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

import torch

import torch.distributed as dist


def gedi_nogen_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, beta: float = 0.05
) -> torch.Tensor:
    """Computes GEDI nogen' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        beta (float, optional): beta factor of the loss. Defaults to 0.01.

    Returns:
        torch.Tensor: GEDI nogen' loss.
    """

    # First loss term (UNIFORMITY)
    loss1 = - torch.log(z1.softmax(1).mean(0)).sum()
    
    # Second loss term (INVARIANCE)
    loss2 = - (z2.softmax(1) * z1).sum(1).mean() + z1.logsumexp(1).mean()

    loss = 0.5 * loss1 + 0.5 * beta * loss2

    return loss
