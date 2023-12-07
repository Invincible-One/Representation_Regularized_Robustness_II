import os
import sys
import csv


import numpy as np

import torch

from utils.helper import progress_bar


def log_args(args):
    max_len = max(len(arg_name) for arg_name in vars(args))
    line = '*' * (max_len + 2 + 20)

    print(line)

    print("Arguments & Hyperparameters".center(max_len + 2 + 20))

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name.ljust(max_len)}: {arg_value}")

    print(line)


def log_results(acc, loss, split):
    return f"[{split}] acc {acc:.4f} loss {loss:.4f}"
