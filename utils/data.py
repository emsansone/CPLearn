"""
Utilities for data.
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def logit(x, alpha=0.):
    x = x * (1. - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1. - x)

class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))

def get_data(args):
    """
    Get data.
    """
    if args.unit_interval:
        post_trans = [lambda x: x]
        post_trans_inv = lambda x: x
    elif args.logit:
        post_trans = [lambda x: (x + (torch.rand_like(x) - 0.5) / 256).clamp(1e-3, 1-1e-3), logit]
        post_trans_inv = lambda x: x.sigmoid()
    else:
        if args.dataset == "mnist":
            # post_trans = [lambda x: 2 * x - 1]
            # post_trans_inv = lambda x: (x + 1) / 2
            MEAN, STD = 0.1, 0.28 # STATISTICS COMPUTED FROM THE DATASET run python3 utils/data.py to check
            post_trans = [lambda x: (x - MEAN) / STD]
            post_trans_inv = lambda x: (x * STD) + MEAN
        elif args.dataset == "svhn":
            MEAN, STD = 0.45, 0.19 # STATISTICS COMPUTED FROM THE DATASET run python3 utils/data.py to check
            post_trans = [lambda x: (x - MEAN) / STD]
            post_trans_inv = lambda x: (x * STD) + MEAN
        elif args.dataset == "cifar10":
            MEAN, STD = 0.48, 0.25 # STATISTICS COMPUTED FROM THE DATASET run python3 utils/data.py to check
            post_trans = [lambda x: (x - MEAN) / STD]
            post_trans_inv = lambda x: (x * STD) + MEAN
        elif args.dataset == "cifar100":
            MEAN, STD = 0.48, 0.27 # STATISTICS COMPUTED FROM THE DATASET run python3 utils/data.py to check
            post_trans = [lambda x: (x - MEAN) / STD]
            post_trans_inv = lambda x: (x * STD) + MEAN


    gauss = [lambda x: x + args.sigma * torch.randn_like(x)]

    if args.dataset in ["mnist"]:
        if args.data_aug:
            augs = [transforms.Grayscale(3),
                    transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomApply(
                        [transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomCrop(32)]
            print("using data augmentation")
        else:
            #augs = [transforms.Pad(4, padding_mode="reflect"), 
            #        transforms.RandomCrop(32)]
            augs = [transforms.Grayscale(3)]

        transf = MultiSample(
                    transforms.Compose(augs +
                                    [transforms.ToTensor()] +
                                    post_trans +
                                    gauss)
                )

        tr_dataset = datasets.MNIST("./data",
                                    transform=transf,
                                    download=True)
        te_dataset = datasets.MNIST("./data", train=False,
                                    transform=transf,
                                    download=True)

        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))

        return tr_dload, te_dload, plot

    elif args.dataset == "svhn":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"), 
                    transforms.RandomApply(
                        [transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomGrayscale(p=args.gray_scale_prob),
                    transforms.RandomCrop(32)]
            print("using data augmentation")
        else:
            augs = []

        transf = MultiSample(
                    transforms.Compose(augs +
                                    [transforms.ToTensor()] +
                                    post_trans +
                                    gauss)
                )
        tr_dataset = datasets.SVHN("./data",
                                    transform=transf,
                                    download=True)
        te_dataset = datasets.SVHN("./data", split='test',
                                    transform=transf,
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    elif args.dataset == "cifar10":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomApply(
                        [transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomGrayscale(p=args.gray_scale_prob),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip()]
            print("using data augmentation")
        else:
            augs = []

        transf = MultiSample(
                    transforms.Compose(augs +
                                    [transforms.ToTensor()] +
                                    post_trans +
                                    gauss)
                )
        tr_dataset = datasets.CIFAR10("./data",
                                      transform=transf,
                                      download=True)
        te_dataset = datasets.CIFAR10("./data", train=False,
                                    transform=transf,
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    elif args.dataset == "cifar100":
        if args.data_aug:
            augs = [transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomApply(
                        [transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomGrayscale(p=args.gray_scale_prob),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip()]
            print("using data augmentation")
        else:
            augs = []

        transf = MultiSample(
                    transforms.Compose(augs +
                                    [transforms.ToTensor()] +
                                    post_trans +
                                    gauss)
                )
        tr_dataset = datasets.CIFAR100("./data",
                                       transform=transf,
                                       download=True)
        te_dataset = datasets.CIFAR100("./data", train=False,
                                    transform=transf,
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(post_trans_inv(x), p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    else:
        raise NotImplementedError


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("Experiments for GEDI")

    # method
    parser.add_argument("--method", type=str, default="gedi", help="jem, swav, gedi1, gedi2, gedi3, gedi4")

    # data
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument('--brightness', type=float, default=0.4, help="Data augmentation: color jitter (brightness)")
    parser.add_argument('--contrast', type=float, default=0.4, help="Data augmentation: color jitter (contrast)")
    parser.add_argument('--saturation', type=float, default=0.4, help="Data augmentation: color jitter (saturation)")
    parser.add_argument('--hue', type=float, default=0.1, help="Data augmentation: color jitter (hue)")
    parser.add_argument('--color_jitter_prob', type=float, default=0.1, help="Data augmentation: color jitter (probability)")
    parser.add_argument('--gray_scale_prob', type=float, default=0.1, help="Data augmentation: Gray scale (probability)")

    parser.add_argument("--dataset", type=str, default="svhn",
                                choices=["svhn", "cifar10", "cifar100", "mnist"])
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--unit_interval", action="store_true")
    parser.add_argument("--logit", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument('--img_size', type=int)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    SEED = 1234
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # data
    train_loader, _, plot = get_data(args)

    for x_d, y_d in train_loader:
        x_d = x_d[0]
        args.data_aug = False
        DATA = 0
        print(args.data_aug)
        print('Shape', x_d.shape)
        print('Variance {}; Mean {}'.format(*torch.var_mean(x_d, dim=(0,2,3))))
        
        break
