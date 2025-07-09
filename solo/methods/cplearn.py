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

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.cplearn import cplearn_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import numpy as np


class CPLearn(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements CPLearn

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                beta (float): beta of the loss.
        """

        super().__init__(cfg)

        self.beta: float = cfg.method_kwargs.beta
        self.epsilon = 1e-8

        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        print('||||||||||||||||||||||||||||||||||||||||||||||||||||')
        print(self.beta, self.proj_output_dim)
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||')

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
        )
        self.tanh = nn.Tanh()
        self.register_buffer('weights', torch.tensor(2. * np.random.randint(2, size=(self.proj_hidden_dim, self.proj_output_dim)) - 1.).to(torch.float))


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CPLearn, CPLearn).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.beta")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        z = self.tanh(z)
        # z = F.normalize(z)
        z = z @ self.weights
        n, c = z.shape
        temp = self.proj_hidden_dim / (np.sqrt(n) * np.log((1. - self.epsilon * (c - 1.)) / self.epsilon))
        out.update({"z": z / temp})
        return out

    def embeddings(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = super().forward(X)
        z = out["feats"]
        for layer in self.projector:
            z = layer(z)
        z = self.tanh(z)
        out.update({"embed": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for CPLearn reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of CPLearn loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- CPLearn loss -------
        cplearn_loss = cplearn_loss_func(z1, z2, beta=self.beta)

        self.log("train_cplearn", cplearn_loss, on_epoch=True, sync_dist=True)

        return cplearn_loss + class_loss
