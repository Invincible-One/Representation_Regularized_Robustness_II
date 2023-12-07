import os

import torch
import torch.nn as nn

from data import data_config
from models.custom_models import FullyConnectedNetwork, VGG_like



model_config = {
    "FullyConnectedNetwork": {
        "input_type": "flattened",
    },
    "VGG_like": {
        "input_type": "image",
    },
}


def get_model(args):
    input_dim = data_config[args.dataset]["input_dim"]
    output_dim = data_config[args.dataset]["output_dim"]
    if input_dim is None:
        input_dim = args.input_dim
    
    if args.model == "FullyConnectedNetwork":
        network = FullyConnectedNetwork(
                input_dim=input_dim,
                config=args.model_config,
                output_dim=output_dim,
                )
    elif args.model == "VGG_like":
        network = VGG_like(
                input_dim=input_dim,
                config=args.model_config,
                output_dim=output_dim,
                )
    else:
        raise NotImplementedError

    return network



def save_model(args, model):
    base_folder = "/scratch/ym2380/saved_models"
    fname = f"{args.exp_name}.pth"

    parent_folder = os.path.join(base_folder, args.model, args.dataset)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    fpath = os.path.join(parent_folder, fname)

    model.to('cpu')
    torch.save(model.state_dict(), fpath)
