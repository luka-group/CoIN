import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import set_seed


def load_contrastive_dataset(datafile_path: str):
    if "csv" in datafile_path:
        data = pd.read_csv(datafile_path)
    else:
        with open(datafile_path, 'rb') as data_f:
            data = pickle.load(data_f)
    if type(data) == pd.DataFrame:
        return Dataset.from_pandas(data)
    else:
        return data


def generate_flan_input(inputs, targets=None):
    res = f"{inputs}\n\n### Response:\n"
    if targets:
        res = f"{res}{targets}"
    return res


def output_args_to_pkl(args: dict, output_dir: str, filepath: str):
    Path(os.path.join(output_dir)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, filepath), "wb") as args_out:
        pickle.dump(args, args_out)


def output_args_to_txt(args: dict, output_dir: str, filepath: str):
    Path(os.path.join(output_dir)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, filepath), "w") as file:
        for key, value in args.items():
            file.write(f'{key}: {value}\n')


def load_data_from_pkl(output_dir: str, filepath: str):
    with open(os.path.join(output_dir, filepath), "rb") as args_in:
        return pickle.load(args_in)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
