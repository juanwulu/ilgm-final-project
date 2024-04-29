# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Multivariate Student-t distributions implemented in PyTorch.

The code is adapted from the following source: https://github.com/pyro-
ppl/pyro/blob/dev/pyro/distributions/multivariate_studentt.py
"""
from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch
from torch import NumberType, Size, Tensor
from torch.distributions import Chi2, Distribution, constraints
from torch.distributions.utils import lazy_property

__all__ = ["MultivariateStudent"]


class MultivariateStudent(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        df: Union[NumberType, Tensor],
        loc: Tensor,
        scale: Optional[Tensor] = None,
        scale_tril: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Multivariate Student-t distribution.

        Args:
            df (Tensor): Degrees of freedom.
            loc (Tensor): Mean.
            scale_tril (Tensor): Lower Cholesky factor of covariance matrix.
            validate_args (Optional[bool], optional): Whether to validate input
                with asserts. Defaults to None.
        """
        if (scale is None) and (scale_tril is None):
            raise ValueError(
                "Either `scale` or `scale_tril` must be specified."
            )
        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        if scale is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(scale)
        dim = loc.size(-1)
        assert self._unbroadcasted_scale_tril.shape[-2:] == (dim, dim)
        batch_shape = torch.broadcast_shapes(
            df.shape, loc.shape[:-1], self._unbroadcasted_scale_tril.shape[:-2]
        )
        event_shape = Size((dim,))
        self.df = df.expand(batch_shape)
        self.loc = loc.expand(batch_shape + event_shape)
        self._chi2 = Chi2(self.df)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Tensor: Mean."""
        m = self.loc.clone()
        m[self.df <= 1, :] = float("nan")
        return self.loc

    @property
    def variance(self) -> Tensor:
        """Tensor: Variance."""
        m = self.scale_tril.pow(2).sum(-1) * (
            self.df / (self.df - 2)
        ).unsqueeze(-1)
        m[(self.df <= 2) & (self.df > 1), :] = float("inf")
        m[self.df <= 1, :] = float("nan")
        return m

    @lazy_property
    def scale_tril(self) -> Tensor:
        """Tensor: Lower Cholesky factor of covariance matrix."""
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        """Tensor: Covariance matrix."""
        return torch.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(-1, -2),
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self) -> Tensor:
        """Tensor: Precision matrix."""
        eye = torch.eye(
            self.loc.size(-1),
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return torch.cholesky_solve(
            eye, self._unbroadcasted_scale_tril
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    def expand(
        self, batch_shape: Size, _instance: Optional[Any] = None
    ) -> "MultivariateStudent":
        new = self._get_checked_instance(MultivariateStudent, _instance)
        batch_shape = Size(batch_shape)
        loc_shape = batch_shape + self._event_shape
        scale_shape = loc_shape + self._event_shape
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(scale_shape)
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(scale_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(scale_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(MultivariateStudent, new).__init__(
            batch_shape, self._event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        n = self.loc.size(-1)
        diff = value - self.loc
        y: Tensor = torch.linalg.solve_triangular(
            self.scale_tril, diff.unsqueeze(-1), upper=False
        )
        y = y.squeeze(-1).pow(2).sum(-1)
        Z: Tensor = (
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + 0.5 * n * self.df.log()
            + 0.5 * n * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + n))
        )
        return -0.5 * (self.df + n) * torch.log1p(y / self.df) - Z

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        X = torch.empty(
            shape, dtype=self.loc.dtype, device=self.loc.device
        ).normal_()
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + torch.matmul(
            self.scale_tril, Y.unsqueeze(-1)
        ).squeeze(-1)
