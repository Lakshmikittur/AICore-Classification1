import torch


def get(args, parameters):
    return getattr(torch.optim, args.optimizer)(parameters, lr=args.lr, weight_decay = 0.01)
#0.0001 0.46200
#0.001 0.46270
#0.01 0.45200