import re

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.transforms import GrayToRGB, NumpyToPIL
from data.datasets import TwilightDuo, ColoredMNIST
#from models import model_config



data_config = {
        "Synthetic": {
            "data_type": None,
            "input_dim": None,
            "output_dim": None,
            },
        "TwilightDuo": {
            "data_type": "image",
            "input_dim": (1, 200, 200),
            "output_dim": 2,
            },
        "ColoredMNIST": {
            "data_type": "image",
            "input_dim": (3, 28, 28),
            "output_dim": 2,
            },
        }



def get_data(args, split):
    if args.dataset == "Synthetic":
        raise NotImplementedError("Synthetic data is deleted!")

    elif args.dataset == "TwilightDuo":
        assert split in {'train', "val1", "val2"}, "Exceptional split!"
        if args.model == "resnet50":
            transform = transforms.Compose([
                GrayToRGB(),
                NumpyToPIL(),
                #WARNING: not general. did this to avoid the circular import error
                transforms.Resize((224, 224)),
                #transforms.Resize(model_config[args.model]["target_resolution"]),
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.ToTensor()

        if split == 'train':
            n_samples = args.train_size
            bias = args.train_bias
        elif split == "val1":
            n_samples = args.val_size
            bias = args.train_bias
        else:   # split == "val2"
            n_samples = args.val_size
            bias = args.val_bias

        data = TwilightDuo(n_samples=n_samples, bias=bias, transform=transform)

    elif args.dataset == "ColoredMNIST":
        if args.model == "resnet50":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0),
                    (0.3801, 0.3801, 0.3801)),
                transforms.Resize((224, 224))
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307, 0.1307, 0),
                    (0.3801, 0.3801, 0.3801)),
                ])

        split_pattern = r"^val(\d+)$"
        split_match = re.fullmatch(split_pattern, split)
        if split_match:
            train = False
            val_idx = int(split_match.group(1)) - 1
            mnist_corr = args.val_mnist_corr[val_idx]
            color_corr = args.val_color_corr[val_idx]
        elif split == 'train':
            train = True
            mnist_corr = args.train_mnist_corr
            color_corr = args.train_color_corr
        else:
            raise Exception("Exceptional split!")

        data = ColoredMNIST(
                root="/scratch/ym2380/data/",
                train=True,
                mnist_corr=mnist_corr,
                color_corr=color_corr,
                transform=transform,
                )

    else:
        raise NotImplementedError

    return data



def get_loader(args, dataset, split):
    loader_kwargs = {
            "batch_size": args.batch_size if split == 'train' else args.val_batch_size,
            "num_workers": args.n_workers,
            "pin_memory": True,
            }
    
    if args.dataset == "Synthetic":
        raise NotImplementedError("Synthetic data is deleted!")

    elif args.dataset == "TwilightDuo":
        assert split in {'train', "val1", "val2"}, "Exceptional split!"
        if split == 'train':
            loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
        else: ###split in {"val1", "val2"}
            loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

    elif args.dataset == "ColoredMNIST":
        split_pattern = r"^val(\d+)$"
        split_match = re.fullmatch(split_pattern, split)
        if split_match:
            loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
        elif split == 'train':
            loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
        else:
            raise Exception("Exceptional split!")

    else:
        raise NotImplementedError

    return loader


