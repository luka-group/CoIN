from typing import Optional, Union, Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.utils import PaddingStrategy


class ContrastiveDataCollator:
    """
    For collating contrastive data
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 model: Optional[Any] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100,
                 return_tensors: str = "pt",
                 pad_to_left: bool = True
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.pad_to_left = pad_to_left

    def get_collated_result(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return features

    def separate_prompts(self, input_ids: list, attention_mask: list, labels: list):
        original = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "labels": labels[0]
        }
        paraphrased = {
            "input_ids": input_ids[1],
            "attention_mask": attention_mask[1],
            "labels": labels[1]
        }
        return original, paraphrased

    def __call__(self, batch, return_tensors=None):
        """
        batch: (batch_size, entry)
        each entry: {
            'input_ids': [[original], [paraphrased]],
            'attention_mask': [[original], [paraphrased]]
            'labels': [[original], [paraphrased]]
        }
        """
        all_original_tokenized_full_prompt = []
        all_paraphrased_tokenized_full_prompt = []

        for entry in batch:
            original_tokenized_full_prompt, paraphrased_tokenized_full_prompt = self.separate_prompts(
                                                                                        entry["input_ids"],
                                                                                        entry["attention_mask"],
                                                                                        entry["labels"]
                                                                                    )
            all_original_tokenized_full_prompt.append(original_tokenized_full_prompt)
            all_paraphrased_tokenized_full_prompt.append(paraphrased_tokenized_full_prompt)

        original_collate_results = self.get_collated_result(all_original_tokenized_full_prompt)
        paraphrased_collate_results = self.get_collated_result(all_paraphrased_tokenized_full_prompt)

        return merge_batch_prompts(original_collate_results, paraphrased_collate_results, self.pad_to_left)


def batch_concat(batch1: torch.Tensor, batch2: torch.Tensor, pad_to_left: bool = True, pad_value: int = 0):
    """
    Concatenate 2 batches of tensor together while pad them to max_length among to batches
    """
    max_length = max(batch1.size(1), batch2.size(1))
    batch1_pad_length = max_length - batch1.size(1)
    batch2_pad_length = max_length - batch2.size(1)
    padding1 = torch.ones(batch1.size(0), batch1_pad_length) * pad_value
    padding2 = torch.ones(batch2.size(0), batch2_pad_length) * pad_value
    if pad_to_left:
        padded_batch1 = torch.cat((padding1, batch1), dim=1)
        padded_batch2 = torch.cat((padding2, batch2), dim=1)
    else:
        padded_batch1 = torch.cat((batch1, padding1), dim=1)
        padded_batch2 = torch.cat((batch2, padding2), dim=1)
    return torch.cat((padded_batch1, padded_batch2)), batch1_pad_length, batch2_pad_length


def merge_batch_prompts(original_tokenized_full_prompt: BatchEncoding,
                        paraphrased_tokenized_full_prompt: BatchEncoding, pad_to_left: bool):
    """
    Concat pairs of instructions together (value of each key has shape of (batch_size * 2, ...),
    which will be separated again and processed independently in model's forward function)
    """
    new_input_ids, batch1_pad_length, batch2_pad_length = batch_concat(original_tokenized_full_prompt["input_ids"], paraphrased_tokenized_full_prompt["input_ids"], pad_to_left)
    new_attention_mask, _, _ = batch_concat(original_tokenized_full_prompt["attention_mask"], paraphrased_tokenized_full_prompt["attention_mask"], pad_to_left)
    new_labels, _, _ = batch_concat(original_tokenized_full_prompt["labels"], paraphrased_tokenized_full_prompt["labels"], pad_to_left, pad_value=-100)
    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels,
    }


def separate_batch_prompts(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """
    Separate merged pair of prompts to original batch & paraphrased batch
    """
    original_tokenized_full_prompt = {
        "input_ids": torch.squeeze(input_ids[:batch_size], dim=1).long(),
        "attention_mask": torch.squeeze(attention_mask[:batch_size], dim=1),
        "labels": torch.squeeze(labels[:batch_size], dim=1).long()
    }

    paraphrased_tokenized_full_prompt = {
        "input_ids": torch.squeeze(input_ids[batch_size:], dim=1).long(),
        "attention_mask": torch.squeeze(attention_mask[batch_size:], dim=1),
        "labels": torch.squeeze(labels[batch_size:], dim=1).long()
    }

    return original_tokenized_full_prompt, paraphrased_tokenized_full_prompt
