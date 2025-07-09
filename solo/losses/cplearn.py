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


def cplearn_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, beta: float = 0.05
) -> torch.Tensor:
    """Computes CPLearn' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        beta (float, optional): beta factor of the loss. Defaults to 0.01.

    Returns:
        torch.Tensor: CPLearn' loss.
    """

    # ##### Single loss code ######
    z1 = z1 - torch.max(z1, dim=1, keepdim=True)[0]
    z2 = z2 - torch.max(z2, dim=1, keepdim=True)[0]
    _, c = z1.shape
    loss1 = ( z2.softmax(1) * torch.log( c * z2.softmax(1).mean(0, keepdim=True)) ).sum(1).mean()
    loss2 = - ( z2.softmax(1) * (beta * z1) - z2.softmax(1) * (beta * z1.logsumexp(1, keepdim=True)) ).sum(1).mean()

    loss3 = ( z1.softmax(1) * torch.log( c * z1.softmax(1).mean(0, keepdim=True)) ).sum(1).mean()
    loss4 = - ( z1.softmax(1) * (beta * z2) - z1.softmax(1) * (beta * z2.logsumexp(1, keepdim=True)) ).sum(1).mean()

    # loss5 = ((learn_codes - target_codes)**2).mean() # CALL THIS TO AVOID PROBLEM OF UNUSED PARAMETERS

    # loss6 = (back_out**2).mean()

    loss = 0.5 * (loss1 + loss2) + 0.5 * (loss3 + loss4)

    return loss
