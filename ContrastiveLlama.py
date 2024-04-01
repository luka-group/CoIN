from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

import torch
import wandb
from torch import nn
import torch.nn.functional as f
from torch.nn import CrossEntropyLoss
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput

from ContrastiveDataCollator import separate_batch_prompts


@dataclass
class ContrastiveLlamaOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_original_instruction: torch.FloatTensor = None
    logits_per_paraphrased_instruction: torch.FloatTensor = None
    original_instruction_hidden_states: torch.FloatTensor = None
    paraphrased_instruction_hidden_states: torch.FloatTensor = None
    original_instruction_outputs: BaseModelOutputWithPast = None
    paraphrased_instruction_outputs: BaseModelOutputWithPast = None


class ContrastiveEvalOutput(ModelOutput):
    logits: torch.LongTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = 1 if temperature is None else temperature
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, original_instruction_inputs, paraphrased_instruction_inputs):
        batch_size = original_instruction_inputs.size(0)
        labels = torch.arange(batch_size).to(original_instruction_inputs.device)

        original_to_paraphrased_sim = f.cosine_similarity(original_instruction_inputs.unsqueeze(1),
                                                          paraphrased_instruction_inputs.unsqueeze(0),
                                                          dim=2) / self.temperature
        paraphrased_to_original_sim = original_to_paraphrased_sim.T

        ori_to_para_loss = f.cross_entropy(original_to_paraphrased_sim, labels)
        para_to_ori_loss = f.cross_entropy(paraphrased_to_original_sim, labels)

        loss = (ori_to_para_loss + para_to_ori_loss) / 2

        return loss, original_to_paraphrased_sim, paraphrased_to_original_sim, ori_to_para_loss, para_to_ori_loss


class ContrastiveLlama(LlamaPreTrainedModel):
    def __init__(self, config, do_predict=False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # From original LlamaForCausalLM
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Parameters for Contrastive loss
        self.pooling_method = config.pooling_method
        self.temperature = config.temperature
        self.contrastive_loss = ContrastiveLoss(self.temperature)
        self.contrastive_loss_ratio = config.contrastive_loss_ratio

        self.do_predict = do_predict
        self.do_contrastive = config.do_contrastive

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_decoder_outputs(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def get_entropy_loss_for_token_prediction(self, logits, labels):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_pooled_hidden_states(self, hidden_states):
        """
        Get hidden states of the last token of each sequence (reference: LlamaForSequenceClassification)
        hidden_states: (batch_size, seq_length, vocab_num)
        return: (batch_size, vocab_num)
        """
        if self.pooling_method == 'last':
            return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), -1]
        elif 'average' in self.pooling_method:
            if self.pooling_method == 'average_first_last':
                hidden_states = torch.cat((hidden_states[:, 0], hidden_states[:, -1])).unsqueeze(0)
            if self.pooling_method == 'average_first_last' or self.pooling_method == 'average_all':
                return torch.mean(hidden_states, dim=1)
            else:
                raise ValueError(f"Pooling method {self.self.pooling_method} not supported")
        elif self.pooling_method == 'max':
            return torch.max(hidden_states, dim=1).values
        else:
            raise ValueError(f"Pooling method {self.pooling_metlora_rhod} not supported")

    def get_tensor_except_i(self, input_tensor, i):
        seq_length = input_tensor.size(0)
        if i == 0:
            new_tensor = input_tensor[1:, :, :]
        elif i == seq_length - 1:
            new_tensor = input_tensor[:-1, :, :]
        else:
            left = input_tensor[:i, :, :]
            right = input_tensor[i + 1:, :, :]
            new_tensor = torch.cat((left, right), dim=1)
        return new_tensor

    def scale_contrastive_loss(self, generation_loss, contrastive_loss, max_scale_ratio):
        if contrastive_loss != 0 and contrastive_loss > generation_loss:
            new_contrastive_loss = contrastive_loss * (
                min(max_scale_ratio, generation_loss.detach() / contrastive_loss.detach()))
        else:
            new_contrastive_loss = contrastive_loss
        return new_contrastive_loss

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[ContrastiveLlamaOutput, ContrastiveEvalOutput]:
        if not self.do_predict:
            # Reformat: separate original & paraphrased instructions batch
            original_tokenized_full_prompt, paraphrased_tokenized_full_prompt = separate_batch_prompts(input_ids,
                                                                                                       attention_mask,
                                                                                                       labels,
                                                                                                       int(input_ids.size(
                                                                                                           0) / 2))
            original_outputs = self.get_decoder_outputs(**original_tokenized_full_prompt)
            paraphrased_outputs = self.get_decoder_outputs(**paraphrased_tokenized_full_prompt)

            original_entire_sentence_hidden_states = original_outputs[0]
            paraphrased_entire_sentence_hidden_states = paraphrased_outputs[0]

            original_instruction_logits = self.lm_head(original_entire_sentence_hidden_states)
            paraphrased_instruction_logits = self.lm_head(paraphrased_entire_sentence_hidden_states)

            # Contrastive Loss
            contrastive_loss = 0
            # If it's False: no contrastive loss/continual instruction tuning (ablation)
            if self.do_contrastive:
                contrastive_loss, sent_original_to_paraphrased_sim, sent_inst_paraphrased_to_original_sim, sent_ori_to_para_loss, sent_inst_para_to_ori_loss = self.contrastive_loss(
                    self.get_pooled_hidden_states(original_entire_sentence_hidden_states),
                    self.get_pooled_hidden_states(paraphrased_entire_sentence_hidden_states)
                )

            # Generation loss
            original_instruction_loss = self.get_entropy_loss_for_token_prediction(
                original_instruction_logits,
                original_tokenized_full_prompt["labels"]
            )
            paraphrased_instruction_loss = self.get_entropy_loss_for_token_prediction(
                paraphrased_instruction_logits,
                paraphrased_tokenized_full_prompt["labels"]
            )

            generation_loss = original_instruction_loss + paraphrased_instruction_loss

            contrastive_loss = contrastive_loss * self.contrastive_loss_ratio
            contrastive_loss = self.scale_contrastive_loss(generation_loss, contrastive_loss,
                                                           self.contrastive_loss_ratio)

            loss = contrastive_loss + generation_loss

            wandb.log({
                'total_loss': loss,
                'contrastive_loss': contrastive_loss,
                'generation_loss': generation_loss,
                'original_generation_loss': original_instruction_loss,
                'paraphrased_generation_loss': paraphrased_instruction_loss
            })

            return ContrastiveLlamaOutput(
                loss=loss,
                original_instruction_hidden_states=original_entire_sentence_hidden_states,
                paraphrased_instruction_hidden_states=paraphrased_entire_sentence_hidden_states,
                original_instruction_outputs=original_outputs,
                paraphrased_instruction_outputs=paraphrased_outputs,
            )
        else:
            # For evaluation (process single inputs instead of pair-wise for contrastive learning)
            outputs = self.get_decoder_outputs(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                labels
            )
            logits = self.lm_head(outputs[0])
            loss = self.get_entropy_loss_for_token_prediction(logits, labels)
            return ContrastiveEvalOutput(
                loss=loss,
                logits=logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states
            )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
