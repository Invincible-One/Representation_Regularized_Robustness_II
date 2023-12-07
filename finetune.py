import os
import argparse

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from models import get_model, model_config
from data import data_config
from data.datasets import TwilightDuo
from main import get_device
from utils.helper import arg_intNchar
from utils.logging import log_args



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices={"FullyConnectedNetwork", "VGG_like", "resnet50"})
    parser.add_argument("--dataset", choices={"Synthetic", "TwilightDuo"})
    parser.add_argument("--model_config", nargs='+', required=True, type=arg_intNchar)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--filename", type=str)
    parser.add_argument("--finetune_layer", type=str, required=True)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--tr_batch_size", type=int, default=25)
    parser.add_argument("--val_batch_size", type=int, default=100)

    args = parser.parse_args()
    return args



def load_model(args, fname):
    if fname is not None:
        base_folder = "/scratch/ym2380/saved_models"   #not general
        fname = f"{fname}.pth"
        fpath = os.path.join(base_folder, args.model, args.dataset, fname)
    
    model = get_model(args)
    
    if fname is not None:
        model.load_state_dict(torch.load(fpath))
    
    for param in model.parameters():
        param.requires_grad = False

    layer = getattr(model, args.finetune_layer)
    for param in layer.parameters():
        param.requires_grad = True
    
    return model


def run_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer,
    train,
    device,
    ):
    if train:
        model.train()
    else:
        network.eval()
    
    total_loss = 0
    correct = 0
    total = len(dataloader.dataset)
    
    with torch.set_grad_enabled(train):
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            if model_config[args.model]["input_type"] == "flattened":
                X = X.view(X.size(0), -1)
            
            if train:
                optimizer.zero_grad()
            
            outputs = model(X)
            loss_v = loss_fn(outputs, y)
            total_loss += loss_v.item()
            
            if train:
                loss_v.backward()
                optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions =  (predicted == y).sum().item()
            correct += correct_predictions
    
    return correct / total, total_loss / total



if __name__ == "__main__":
    #settings
    args = get_args()
    log_args(args)

    device = get_device()

    #data
    train_data = TwilightDuo(n_samples=500, bias=0, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.tr_batch_size, num_workers=1, pin_memory=True)
    val_data = TwilightDuo(n_samples=1000, bias=0, transform=transforms.ToTensor())
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.val_batch_size, num_workers=1, pin_memory=True)

    #model
    network = load_model(args, fname=args.filename).to(device)
    loss_fn = nn.CrossEntropyLoss()

    #optim
    optimizer = optim.Adam(network.classifier.parameters(), lr=args.lr)

    #train
    for i in range(args.n_epochs):
        logs = [f"Epoch {i + 1:02}/{args.n_epochs}", ]
        tr_acc, tr_loss = run_epoch(train_loader, network, loss_fn, optimizer, True, device)
        val_acc, val_loss = run_epoch(val_loader, network, loss_fn, optimizer, False, device)
        tr_log = f"[train] acc {tr_acc:.4f} loss {tr_loss:.4f}"
        val_log = f"[val] acc {val_acc:.4f} loss {val_loss:.4f}"
        logs.append(tr_log)
        logs.append(val_log)
        print(", ".join(logs))
