# Copyright 2025 CPLearn team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
from pathlib import Path
import torch

from omegaconf import OmegaConf

from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS
import argparse
import numpy as np

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_embeddings() -> argparse.Namespace:
    """Parses arguments for analysis of embeddings.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)

    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args


def main(args):

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")]
    assert(len(ckpt_path)==1)
    ckpt_path = ckpt_path[0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    model = METHODS[method_args["method"]](cfg=cfg)
    loaded = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(loaded['state_dict'], strict=False)
    # prepare data
    _, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,
    )

    # move model to the gpu
    if 'SLURM_PROCID' in os.environ:
        procid = int(os.environ['SLURM_PROCID'])
        device = procid % 8
    else:
        device = 0
    model = model.to(device)

    print('\n Extracting Embedding Features\n')
    itr = 0
    for x_d, _ in val_loader:

        x_d = x_d.to(device)

        out = model.embeddings(x_d)
        out = out["embed"].detach().cpu()

        if itr == 0:
            embeddings = out
        else:
            embeddings = torch.cat((embeddings, out), 0)
 
        itr += 1

    os.makedirs('embeddings', exist_ok=True)
    torch.save({'embed': embeddings}, 'embeddings/' + method_args["method"] + '.ckpt')

if __name__ == "__main__":
    mapping = {}
    args = parse_args_embeddings()
    main(args)
