import argparse
import os

import numpy as np
import pandas as pd

sample_cnt_bound = 300
prompt_num = 6

ALL_PERTURB_METHODS = ["clean", "deepwordbug", "textfooler", "checklist", "semantic"]
ALL_INSTRUCTION_OOD_TASKS = ['cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'wnli']

task_category_to_task_list = {
    "sentiment_analysis": ["sst2"],
    "grammar_correctness": ["cola"],
    "duplicate_sentence_detection": ["qqp", "mrpc"],
    "NLI": ["mnli", "mnli_matched", "mnli_mismatched", "qnli", "rte", "wnli"],
}

task_to_category = {
    "sst2": "sentiment_analysis",
    "cola": "grammar_correctness",
    "qqp": "duplicate_sentence_detection",
    "mrpc": "duplicate_sentence_detection",
    "mnli": "NLI",
    "mnli_matched": "NLI",
    "mnli_mismatched": "NLI",
    "qnli": "NLI",
    "rte": "NLI",
    "wnli": "NLI",
}


def calc_avg_std(all_acc):
    avg = np.mean(all_acc)
    std = np.std(all_acc)
    return avg, std


def get_perturb_key_name(perturb):
    return f"{perturb}_acc"


def process_acc(task_list, input_args):
    all_results = {}
    for task in task_list:
        all_results.setdefault(task, {f'{get_perturb_key_name(perturb)}': "-" for perturb in ALL_PERTURB_METHODS})
        all_results.setdefault(f"{task}_std",
                               {f'{get_perturb_key_name(perturb)}': "-" for perturb in ALL_PERTURB_METHODS})
    all_results["avg"] = {f'{get_perturb_key_name(perturb)}': "-" for perturb in ALL_PERTURB_METHODS}
    all_results["avg_std"] = {f'{get_perturb_key_name(perturb)}': "-" for perturb in ALL_PERTURB_METHODS}

    for perturb in ALL_PERTURB_METHODS:
        for task in task_list:
            all_acc = []
            for prompt_idx in range(prompt_num):
                task_result_df_path = os.path.join(input_args.output_dir, "results",
                                              f"{task}/{perturb}/len_7_0_shot_prompt{prompt_idx}.csv")
                if not os.path.exists(task_result_df_path):
                    print(f"File {task_result_df_path} does not exist. Skipping...")
                    continue
                df = pd.read_csv(task_result_df_path).iloc[:sample_cnt_bound]
                if type(df['pred'][0]) == str:
                    df['pred'] = df['pred'].apply(lambda x: x.lower())
                    df['gt'] = df['gt'].apply(lambda x: x.lower())
                acc = sum(df['pred'] == df['gt']) / len(df) * 100
                all_acc.append(acc)
            avg, std = calc_avg_std(all_acc)
            all_results[task][get_perturb_key_name(perturb)] = f"{avg:.2f}"
            all_results[f"{task}_std"][get_perturb_key_name(perturb)] = f"{std:.2f}"
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(input_args.output_dir, f"unseen_instruction_acc.csv"))


def main(args):
    process_acc(ALL_INSTRUCTION_OOD_TASKS, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="where the outputs of the model are stored", required=True)

    args = parser.parse_args()

    main(args)
