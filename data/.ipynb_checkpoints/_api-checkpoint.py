import torch

from torch.utils.data import DataLoader

from data.datasets import SyntheticData



def get_data(args, split):
    if args.dataset == "Synthetic":
        assert split in {"train", "val", "val1", "val2"}
        
        d1 = args.cause_dim
        d2 = args.shortcut_dim
        d3 = args.input_dim - d1 - d2
        p = args.p
        q = args.q
        theta1 = args.theta1
        theta3 = args.theta3

        if split in {"train", "val"}:
            thetay = args.thetay_train
            theta2 = args.theta2_train
            if split == "train":
                n_samples = args.train_size
            else:   ###split == "val"
                n_samples = args.val_size
        else:   ###split in {"val1", "val2"}
            n_samples = args.val_size
            if split == "val1":
                thetay = args.thetay_val1
                theta2 = args.theta2_val1
            else:   ###split == "val2"
                thetay = args.thetay_val2
                theta2 = args.theta2_val2

        data = SyntheticData(
                n_samples = n_samples,
                d1 = d1,
                d2 = d2,
                d3 = d3,
                theta1 = theta1,
                theta2 = theta2,
                theta3 = theta3,
                thetay = thetay,
                p = p,
                q = q,
                )
    else:
        raise NotImplementedError

    return data



def get_loader(args, dataset, split):
    loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.n_workers,
            "pin_memory": True,
            }
    if args.dataset == "Synthetic":
        assert split in {"train", "val", "val1", "val2"}
        if split == 'train':
            loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
        else: ###split in {'val', "val1", "val2"}
            loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
    else:
        raise NotImplementedError

    return loader


