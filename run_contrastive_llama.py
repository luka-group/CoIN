import argparse
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch
import transformers
import wandb

from ContrastiveDataCollator import ContrastiveDataCollator
from ContrastiveLlama import ContrastiveLlama
from ContrastiveTrainer import ContrastiveTrainer
import promptbench.main as promptbench_main

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, PeftModel,
)
from transformers import LlamaTokenizer, LlamaConfig, HfArgumentParser, GenerationConfig, \
    LlamaForCausalLM, set_seed

from promptbench.postprocessing import ALL_INSTRUCTION_OOD_TASKS
from utils.common_utils import load_contrastive_dataset, generate_flan_input, output_args_to_pkl, output_args_to_txt, \
    seed_all, load_data_from_pkl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class UnseenInstructionEvalArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # Dataset src params
    promptbench_eval_task: str = field(default='all',
                                       metadata={"help": "Name of task on which model is evaluated",
                                                 "choices": ['all', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched',
                                                             'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'wnli']},
                                       )
    perturb_method: Optional[str] = field(
        default="all",
        metadata={
            "help": "Perturbation method for perturbing unseen instructions.",
            "choices": ["all", "clean", "deepwordbug", "textfooler", "checklist", "semantic"]
            # Corresponds to clean, character, word, sentence, and semantic
        }
    )
    eval_sample_cnt: Optional[int] = field(
        default=300,
        metadata={
            "help": "Number of samples to evaluate"
        },
    )


@dataclass
class ContrastiveLlamaTrainingArgument:
    # Model and lora hyperparams
    base_model: Optional[str] = field(
        default='yahma/llama-7b-hf',
        metadata={"help": "Base model of Llama"}
    )
    lora_weights: Optional[str] = field(
        default="tloen/alpaca-lora-7b",
        metadata={
            "help": "lora weights that will be loaded to peft model (when not using contrastive loss)",
            "choices": [None, "tloen/alpaca-lora-7b", "chansung/alpaca-lora-13b"]
        }
    )
    lora_r: Optional[int] = field(
        default=16
    )
    lora_alpha: Optional[int] = field(
        default=16
    )
    lora_dropout: Optional[float] = field(
        default=0.05
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory=list
    )

    # Data
    use_contrastive_data: Optional[bool] = field(
        default=True,
        metadata={"help": "True for running ablation studies and training contrastive alpaca."}
    )
    data_path: Optional[str] = field(
        default='dataset/contrastive_flan_data.csv',
    )
    add_eos_token: Optional[bool] = field(
        default=False
    )
    output_dir: Optional[str] = field(
        default='./outputs',
    )
    cache_dir: Optional[str] = field(
        default=None,
    )

    # Training args
    do_train: Optional[bool] = field(
        default=False,
    )
    do_predict: Optional[bool] = field(
        default=False,
        metadata={"help": "Run evaluation on unseen instructions"}
    )
    seed: Optional[int] = field(
        default=42,
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "directory of the saved adapter model"}
    )
    batch_size: Optional[int] = field(
        default=64
    )
    micro_batch_size: Optional[int] = field(
        default=2
    )
    num_epochs: Optional[int] = field(
        default=1
    )
    group_by_length: Optional[bool] = field(
        default=False,
        metadata={"help": "faster, but produces an odd training loss curve"}
    )
    learning_rate: Optional[float] = field(
        default=1e-4
    )
    cutoff_len: Optional[int] = field(
        default=256
    )
    val_set_size: Optional[int] = field(
        default=2000
    )

    # Contrastive loss params
    do_contrastive: Optional[bool] = field(
        default=True,
        metadata={
            "help": "True for training with contrastive loss. False for continual instruction tuning."}
    )
    temperature: Optional[float] = field(
        default=0.05,
        metadata={"help": "temperature for contrastive loss"}
    )
    contrastive_loss_ratio: Optional[float] = field(
        default=1000,
        metadata={
            "help": "weight assigned to contrastive loss"}
    )
    pooling_method: Optional[str] = field(
        default='last',
        metadata={
            "help": "How to pool hidden states for contrastive learning. Choices: ['average_all', 'average_first_last', 'max', 'last']"}
    )

    # wandb params
    wandb_usr: Optional[str] = field(
        default=None
    )
    wandb_project: Optional[str] = field(
        default=""
    )
    wandb_run_name: Optional[str] = field(
        default=""
    )
    wandb_watch: Optional[str] = field(
        default="",
        metadata={"help": "options: false | gradients | all"}
    )
    wandb_log_model: Optional[str] = field(
        default="",
        metadata={"help": "options: false | true"}
    )


def initialize_pretrained_peft_model(contrastive_llama_args, device_map):
    # Load pretrained lora weights to the Contrastive peft model
    pretrained_llama_model = LlamaForCausalLM.from_pretrained(
        contrastive_llama_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=contrastive_llama_args.cache_dir
    )
    print(f"Initialize with lora weights from {contrastive_llama_args.lora_weights} to contrastive model")
    logger.info(
        f"Initialize with lora weights from {contrastive_llama_args.lora_weights} to contrastive model")
    pretrained_peft_model = PeftModel.from_pretrained(
        pretrained_llama_model,
        contrastive_llama_args.lora_weights,
        device_map=device_map,
        torch_dtype=torch.float16,
        cache_dir=contrastive_llama_args.cache_dir
    )
    return pretrained_peft_model


def main():
    parser = HfArgumentParser((ContrastiveLlamaTrainingArgument, UnseenInstructionEvalArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        contrastive_llama_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        contrastive_llama_args, eval_args = parser.parse_args_into_dataclasses()
        if contrastive_llama_args.do_train:
            # Save arguments
            output_args_to_pkl(contrastive_llama_args, contrastive_llama_args.output_dir, "contrastive_llama_args.pkl")
            output_args_to_txt(asdict(contrastive_llama_args), contrastive_llama_args.output_dir,
                               "contrastive_llama_args.txt")
            output_args_to_pkl(eval_args, contrastive_llama_args.output_dir, "eval_args.pkl")
            output_args_to_txt(asdict(eval_args), contrastive_llama_args.output_dir, "eval_args.txt")

    seed_all(contrastive_llama_args.seed)

    if contrastive_llama_args.do_train:
        logger.info(f"Run with train params:\n {contrastive_llama_args}")
    if contrastive_llama_args.do_predict:
        logger.info(f"Run with predict params:\n {eval_args}")

    gradient_accumulation_steps = contrastive_llama_args.batch_size // contrastive_llama_args.micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(contrastive_llama_args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(contrastive_llama_args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = contrastive_llama_args.wandb_project
    if len(contrastive_llama_args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = contrastive_llama_args.wandb_watch
    if len(contrastive_llama_args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = contrastive_llama_args.wandb_log_model

    # -------- Tokenizer & functions for tokenizing pair-wise instructions ----
    tokenizer = LlamaTokenizer.from_pretrained(contrastive_llama_args.base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(original_prompt: str, paraphrased_prompt: str, add_eos_token=False):

        result = tokenizer(
            (original_prompt, paraphrased_prompt),
            truncation=True,
            max_length=contrastive_llama_args.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        for idx in range(2):
            # idx: 0 for original, 1 for paraphrased
            if (
                    result["input_ids"][idx][-1] != tokenizer.eos_token_id
                    and len(result["input_ids"][idx]) < contrastive_llama_args.cutoff_len
                    and add_eos_token
            ):
                result["input_ids"][idx].append(tokenizer.eos_token_id)
                result["attention_mask"][idx].append(1)

        result["labels"] = result["input_ids"].copy()
        return result, tokenizer.eos_token_id


    def generate_and_tokenize_prompt(data_point):
        """
        data_point: original_instruction, input, output, paraphrased_instruction
        Use pad token (in this case <unk>) as separator between 2 instructions for concatenation
        """
        original_instruction = data_point['original_instruction']
        paraphrased_instruction = data_point['paraphrased_instruction']

        original_full_prompt = generate_flan_input(
            original_instruction,
            data_point["targets"],
        )
        paraphrased_full_prompt = generate_flan_input(
            paraphrased_instruction,
            data_point["targets"],
        )
        tokenized_full_prompt, eos_token = tokenize(original_full_prompt, paraphrased_full_prompt, add_eos_token=contrastive_llama_args.add_eos_token)
        original_user_prompt = generate_flan_input(original_instruction)
        paraphrased_user_prompt = generate_flan_input(paraphrased_instruction)
        tokenized_user_prompt, eos_token = tokenize(original_user_prompt, paraphrased_user_prompt, add_eos_token=contrastive_llama_args.add_eos_token)
        for i in range(2):
            # 0 for original 1 for paraphrased
            tmp_input_ids = tokenized_user_prompt["input_ids"][i]
            user_prompt_len = len(tmp_input_ids)
            if contrastive_llama_args.add_eos_token and tmp_input_ids[-1] == eos_token:
                user_prompt_len -= 1
            # For not train on inputs
            tokenized_full_prompt["labels"][i] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][i][
                                                                            user_prompt_len:]

        return tokenized_full_prompt

    # -------- Prepare config and base model --------
    if contrastive_llama_args.use_contrastive_data:
        model_config = LlamaConfig.from_pretrained(contrastive_llama_args.base_model)

        # Set parameter for contrastive loss
        model_config.base_model = contrastive_llama_args.base_model
        model_config.temperature = contrastive_llama_args.temperature
        model_config.device_map = device_map
        model_config.contrastive_loss_ratio = contrastive_llama_args.contrastive_loss_ratio
        model_config.do_contrastive = contrastive_llama_args.do_contrastive
        model_config.pooling_method = contrastive_llama_args.pooling_method

        logger.info(f"Initialize ContrastiveLlama w/ base LlamaModel from {model_config.base_model}")
        model = ContrastiveLlama.from_pretrained(
            model_config.base_model,
            config=model_config,
            do_predict=contrastive_llama_args.do_predict,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=model_config.device_map,
            cache_dir=contrastive_llama_args.cache_dir
        )

        model = prepare_model_for_int8_training(model)
        lora_config_args = contrastive_llama_args if contrastive_llama_args.do_train else load_data_from_pkl(
            contrastive_llama_args.resume_from_checkpoint, "contrastive_llama_args.pkl")

        # Prepare peft model for contrastive learning
        config = LoraConfig(
            r=lora_config_args.lora_r,
            lora_alpha=lora_config_args.lora_alpha,
            target_modules=lora_config_args.lora_target_modules,
            lora_dropout=lora_config_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config.base_model = lora_config_args.base_model
        config.temperature = lora_config_args.temperature
        config.device_map = device_map
        model = get_peft_model(model, config)
    else:
        # For running evaluation on Alpaca LoRA
        model = initialize_pretrained_peft_model(contrastive_llama_args, device_map)
        logger.info(f"Initialize Alpaca lora baseline from {contrastive_llama_args.lora_weights}")
        print(f"Initialize Alpaca lora baseline from {contrastive_llama_args.lora_weights}")

    if contrastive_llama_args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            contrastive_llama_args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                contrastive_llama_args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logger.info(f"Starting from {checkpoint_name}")
            print(f"Starting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location=torch.device('cuda'))
            set_peft_model_state_dict(model, adapters_weights)
        else:
            raise ValueError(f"Checkpoint {checkpoint_name} not found")
    else:
        # Load pretrained lora weights to the Contrastive peft model
        pretrained_peft_model = initialize_pretrained_peft_model(contrastive_llama_args, device_map)
        model.base_model.model.model.load_state_dict(pretrained_peft_model.base_model.model.model.state_dict())
        model.base_model.model.lm_head.load_state_dict(pretrained_peft_model.base_model.model.lm_head.state_dict())

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer_args = transformers.TrainingArguments(
        per_device_train_batch_size=contrastive_llama_args.micro_batch_size,
        per_device_eval_batch_size=contrastive_llama_args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=contrastive_llama_args.num_epochs,
        learning_rate=contrastive_llama_args.learning_rate,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps" if contrastive_llama_args.val_set_size > 0 else "no",
        save_strategy="steps",
        output_dir=contrastive_llama_args.output_dir,
        save_total_limit=3,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=contrastive_llama_args.group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=contrastive_llama_args.wandb_run_name if use_wandb else None
    )

    all_metrics = {}

    """ ------------------------ Test on PromptBench and unseen instructions ------------------------"""
    if contrastive_llama_args.do_train:
        data = load_contrastive_dataset(contrastive_llama_args.data_path)
        val_data = None
        if contrastive_llama_args.val_set_size > 0:
            # Not to shuffle the entries to make sure original, positive, and negative samples are in the correct order for contrastive learning.
            train_val = data.train_test_split(test_size=contrastive_llama_args.val_set_size, shuffle=False)
            train_data = train_val["train"].map(generate_and_tokenize_prompt)
            val_data = train_val["test"].map(generate_and_tokenize_prompt)
        else:
            train_data = data.map(generate_and_tokenize_prompt)

        logger.info(f"** Train **")
        # Load training data & initialize trainer
        trainer_args.remove_unused_columns = False
        # warmup steps: 10% of total training steps
        trainer_args.warmup_steps = int(len(train_data) / contrastive_llama_args.batch_size * 0.1)
        eval_steps = int(len(train_data) / contrastive_llama_args.batch_size / 8)
        trainer_args.eval_steps = eval_steps if contrastive_llama_args.val_set_size > 0 else None
        trainer_args.save_steps = eval_steps
        trainer = ContrastiveTrainer(
            model=model,
            args=trainer_args,
            train_dataset=train_data,
            data_collator=ContrastiveDataCollator(tokenizer=tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt", padding=True),
            eval_dataset=val_data,
        )

        model.config.use_cache = False

        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

        # Save model
        model.save_pretrained(contrastive_llama_args.output_dir)

    """ ------------------------ Test on PromptBench and unseen instructions ------------------------"""
    if contrastive_llama_args.do_predict:
        Path(contrastive_llama_args.output_dir).mkdir(parents=True, exist_ok=True)

        # PromptBench
        from promptbench.postprocessing import ALL_PERTURB_METHODS
        attach_method_list = ALL_PERTURB_METHODS if eval_args.perturb_method == 'all' else [eval_args.perturb_method]
        if eval_args.promptbench_eval_task == 'all_instruction_ood':
            promptbench_dataset_list = ALL_INSTRUCTION_OOD_TASKS
        else:
            promptbench_dataset_list = [eval_args.promptbench_eval_task]

        promptbench_args = argparse.Namespace()
        promptbench_args.model = "contrastive_llama"
        promptbench_args.loaded_model = model
        promptbench_args.query_budget = float("inf")
        promptbench_args.verbose = False
        promptbench_args.output_dir = os.path.join(contrastive_llama_args.output_dir, "preds")
        promptbench_args.shot = 0  # all zero-shot
        promptbench_args.prompt_selection = False  # Perturbed versions of all unseen instructions are already created and stored locally. No need to select prompts.
        promptbench_args.sample_cnt = eval_args.eval_sample_cnt
        wandb.init(project=contrastive_llama_args.wandb_project, name=contrastive_llama_args.wandb_run_name)
        for attack_method in attach_method_list:
            for tmp_dataset in promptbench_dataset_list:
                promptbench_args.dataset = tmp_dataset
                promptbench_args.attack = attack_method
                promptbench_args.generate_len = 7
                promptbench_main.main(promptbench_args)
        wandb.finish()


if __name__ == "__main__":
    main()
