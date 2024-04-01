# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging

from peft import PeftModelForCausalLM

from promptbench.dataload import create_dataset
from promptbench.inference import Inference
from promptbench.config import PROMPT_SET_Promptbench_adv as prompt_raw
from promptbench.config import MODEL_SET


def create_logger(log_path):

    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='contrastive_llama', choices=MODEL_SET)     # This project supports contrastive_llama only
    parser.add_argument("--loaded_model", type=PeftModelForCausalLM, default=None, help="Loaded model for contrastive llama eval")
    parser.add_argument('--dataset', type=str, default='bool_logic', choices=["sst2", "cola", "qqp", "mnli", "mnli_matched", "mnli_mismatched", "qnli", "wnli", "rte", "mrpc"])

    parser.add_argument('--query_budget', type=float, default=float("inf"))
    parser.add_argument('--attack', type=str, default='deepwordbug', choices=[
        'textfooler',
        'textbugger',
        'bertattack',
        'deepwordbug',
        'checklist',
        'stresstest',
        'semantic',
        'no', 
        'noattack',
        'clean',
    ])
    parser.add_argument("--verbose", type=bool, default=True)

    parser.add_argument('--output_dir', type=str, default='./')

    parser.add_argument('--model_dir', type=str)

    parser.add_argument('--shot', type=int, default=0)

    parser.add_argument('--generate_len', type=int, default=4)

    parser.add_argument('--prompt_selection', action='store_true')

    # Number of samples to run for evaluation
    parser.add_argument('sample_cnt', type=int, default=300)

    args = parser.parse_args()
    return args


def attack(args, inference_model, RESULTS_DIR):
    dataset_name = args.dataset

    prompt_list = prompt_raw[args.attack][dataset_name]
    for idx, prompt in enumerate(prompt_list):
        acc, result_df = inference_model.predict(prompt)
        args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
        with open(RESULTS_DIR+args.save_file_name+f"_prompt{idx}.txt", "a+") as f:
            f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
        result_df.to_csv(RESULTS_DIR+args.save_file_name+f"_prompt{idx}.csv")


def main(args):
    save_dir = args.dataset

    save_dir += "/"

    LOGS_DIR = os.path.join(args.output_dir, "logs/" + save_dir)
    RESULTS_DIR = os.path.join(args.output_dir, "results/" + save_dir + args.attack + "/")

    for DIR in [LOGS_DIR, RESULTS_DIR]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    file_name = "len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot"

    args.save_file_name = file_name

    data = create_dataset(args.dataset, args.sample_cnt)

    inference_model = Inference(args)
    args.data = data

    logger = create_logger(LOGS_DIR+file_name+".log")
    logger.info(f"attack: {args.attack}, dataset: {args.dataset}\n")
    print(f"attack: {args.attack}, dataset: {args.dataset}\n")

    args.logger = logger

    attack(args, inference_model, RESULTS_DIR)


if __name__ == '__main__':
    args = get_args()
    main(args)
