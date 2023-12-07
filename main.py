import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data import get_data, get_loader
from models import get_model, save_model
from loss import get_loss
from optim import get_optimizer
from train import Runner
from utils.logging import log_args, log_results
from utils.helper import arg_intNchar, arg_floatNchar



def get_args():
    parser = argparse.ArgumentParser()

    #setting
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--log_path", type=str, required=True)

    #data
    parser.add_argument("--dataset", choices={"Synthetic", "TwilightDuo", "ColoredMNIST"})
    
    ##TwilightDuo
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=2000)
    
    parser.add_argument("--train_bias", type=float, default=1)
    parser.add_argument("--val_bias", type=float, default=0)

    ##ColoredMNIST
    parser.add_argument("--train_mnist_corr", type=float, default=1)
    parser.add_argument("--train_color_corr", type=float, default=0.95)
    parser.add_argument("--val_mnist_corr", nargs='+', type=arg_floatNchar, default=[1, 1, 0])
    parser.add_argument("--val_color_corr", nargs='+', type=arg_floatNchar, default=[0.95, 0, 1])

    ##general settings
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=200)

    #loss
    parser.add_argument("--base_loss", choices={"CE",}, default="CE")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)

    parser.add_argument("--repr_module", type=str, required=True)

    #model
    parser.add_argument("--model", choices={"FullyConnectedNetwork", "VGG_like", "resnet50"})
    parser.add_argument("--model_config", nargs='+', required=True, type=arg_intNchar)
    parser.add_argument("--pretrained", action="store_true")
    #parser.add_argument("--output_dim", type=int, default=2)

    #optim
    parser.add_argument("--optimizer", choices={'Adam', 'SGD'}, default='Adam')
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    #run
    parser.add_argument("--n_epochs", type=int, required=True)

    args = parser.parse_args()
    return args


def check_args(args):
    #WARNING: check_args is almost equal to not being implemented
    if args.dataset == "Synthetic":
        raise NotImplementedError("Synthetic data is implemented but not useful. Please don't use it!")
    
    if args.dataset == "TwilightDuo":
        delattr(args, "train_mnist_corr")
        delattr(args, "train_color_corr")
        delattr(args, "val_mnist_corr")
        delattr(args, "val_color_corr")
    
    if args.dataset == "ColoredMNIST":
        assert len(args.val_mnist_corr) == len(args.val_color_corr)
        delattr(args, "train_size")
        delattr(args, "val_size")
        delattr(args, "train_bias")
        delattr(args, "val_bias")



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_splits(args):
    if args.dataset == "TwilightDuo":
        splits = ['train', "val1", "val2"]
    elif args.dataset == "ColoredMNIST":
        splits = ['train',] + [f"val{i + 1}" for i in range(len(args.val_mnist_corr))]
    else:
        raise NotImplementedError

    return splits



if __name__ == "__main__":
    #setting
    args = get_args()
    check_args(args)
    log_args(args)

    device = get_device()

    #data
    splits = get_splits(args)

    datasets = dict()
    dataloaders = dict()
    for split in splits:
        datasets[split] = get_data(args, split=split)
        dataloaders[split] = get_loader(args, dataset=datasets[split], split=split)
    
    #model
    network = get_model(args).to(device)

    #loss
    loss_fn = get_loss(args)

    #optim
    optimizer, scheduler = get_optimizer(args, network)

    #run
    runner = Runner(
            args=args,
            model=network,
            extended_loss=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            repr_module=args.repr_module,
            device=device,
            )
    for i in range(args.n_epochs):
        logs = [f"Epoch {i + 1:02}/{args.n_epochs}", ]
        for split in splits:
            train = True if split == 'train' else False
            runner.update_info(epoch_counter=i + 1, split=split)
            acc, loss = runner.run_epoch(dataloader=dataloaders[split], train=train)
            split_log = log_results(acc=acc, loss=loss, split=split)
            logs.append(split_log)
        print(", ".join(logs))
    runner.remove_hook()
    if args.save_model:
        save_model(args=args, model=network)
