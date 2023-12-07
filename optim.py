import torch
import torch.optim as optim



def get_optimizer(args, model):
    if args.pretrained:
        params = model.fc.parameters()
    else:
        params = model.parameters()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(params=params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler_gamma)
    else:
        scheduler = None
    return optimizer, scheduler
