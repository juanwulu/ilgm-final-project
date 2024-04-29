# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
from __future__ import annotations

import torch


def mvdigamma(value: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes the multivariate digamma function.

    .. note::

        The multivariate digamma function is defined as:

        \\sum_{d=1}^{D}\\psi\\left(\\frac{1}{2}\\left(x_d-d+1\\right)\\right)

    Args:
        value (torch.Tensor): The input tensor of shape `[*, ]`.
        dim (int): The dimension of multivariate digamma function.

    Returns:
        torch.Tensor: The output tensor of shape `[*, ]`.
    """
    assert dim >= 1, ValueError("Invalid dimensionality.")
    assert torch.all(value > (dim - 1) / 2), ValueError("Invalid domain.")
    return torch.sum(
        torch.digamma(
            value.unsqueeze(-1)
            - torch.arange(dim, device=value.device, dtype=value.dtype)
            .div(2)
            .expand(value.shape + (-1,))
        ),
        dim=-1,
    )


def squared_frobenius(value: torch.Tensor) -> torch.Tensor:
    """Computes the squared Frobenius norm of a tensor with arbitrary batches.

    Args:
        value (torch.Tensor): The input tensor of shape `[*, D]`.

    Returns:
        torch.Tensor: The output tensor of shape `[*, ]`.
    """
    m, n = value.size(-2), value.size(-1)
    out = value.reshape(-1, m * n).pow(2).sum(dim=-1)
    out = out.reshape(value.shape[:-2]).contiguous()

    return out
