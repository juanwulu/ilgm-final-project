# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Dataset performance metrics."""
from __future__ import annotations

import torch
from torch_geometric.typing import OptTensor
from torchmetrics import Metric

from .tools import project_poses


class MinAverageDisplacementError(Metric):
    """Compute min-average displacement error for multi-modal predictions."""

    total: torch.Tensor
    """torch.Tensor: total min-average displacement error."""
    count: torch.Tensor
    """torch.Tensor: total number of samples."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Compute and update current minimum average displacement error
        tracker.

        Args:
            preds (torch.Tensor): predicted trajectory of shape `[N, M, T, F]`,
            where `N` is the number of agents, `M` the number of modes for
            multi-modal prediction, `T` the prediction horizon, and `F` the
            number of attributes.
            target (torch.Tensor): ground-truth trajectory. shape: `[N, T, F]`.
        """
        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) >= 2
        ), "Invalid target trajectory!"

        with torch.no_grad():
            preds = preds[..., 0:2].float().to(self.device)
            if preds.ndim == 3:
                # unsqueeze modal dimensionality if necessary
                preds.unsqueeze(1)
            target = target[..., 0:2].float().unsqueeze(1).to(self.device)

            displacement_errors = torch.linalg.norm(
                preds - target, dim=3
            ).mean(2)
            self.total += displacement_errors.min(1)[0].sum()
            self.count += preds.size(0)

    def compute(self) -> torch.FloatTensor:
        return self.total.float() / (self.count + 1e-8)


class MinFinalDisplacementError(Metric):
    """Compute minimum final displacement error for multi-modal predictions."""

    total: torch.Tensor
    """torch.Tensor: total minimum final displacement error."""
    count: torch.Tensor
    """torch.Tensor: total number of samples."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) >= 2
        ), "Invalid target trajectory!"

        with torch.no_grad():
            preds = preds[..., -1, 0:2].float().to(self.device)
            if preds.ndim == 3:
                preds.unsqueeze(1)
            target = target[..., -1, 0:2].float().unsqueeze(1).to(self.device)

            displacement_errors = torch.linalg.norm(preds - target, dim=2)
            self.total += displacement_errors.min(dim=1)[0].sum()
            self.count += preds.size(0)

    def compute(self) -> torch.FloatTensor:
        return self.total.float() / (self.count + 1e-8)


class MissRate(Metric):
    """Compute miss rate for multi-modal predictions."""

    total: torch.Tensor
    """torch.Tensor: total number of miss cases."""
    count: torch.Tensor
    """torch.Tensor: total number of samples."""
    piecewise_lng_threshold: bool
    """bool: whether to use piecewise longitudinal threshold."""

    def __init__(self, piecewise_lng_threshold: bool = True) -> None:
        super().__init__()
        self.piecewise_lng_threshold = piecewise_lng_threshold
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor,
        batch: OptTensor = None,
    ) -> None:
        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        assert preds.size(0) == target.size(0), "Unmatched number of agents!"
        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) >= 2
        ), "Invalid target trajectory!"
        assert (
            anchor.ndim == 2 and anchor.size(-1) == 3
        ), "Invalid coordinate anchors!"

        if batch is None:
            batch = torch.zeros(
                target.size(0), device=self.device, dtype=torch.long
            )
        preds = preds.to(self.device)
        target = target.to(self.device)
        batch = batch.to(self.device)

        if preds.size(-1) == 2:
            # add empty heading dimension if missing from prediction
            with torch.no_grad():
                preds = torch.cat(
                    [
                        preds[..., 0:2],
                        torch.zeros_like(preds[..., 0:1]),
                    ],
                    dim=-1,
                )

        if target.size(-1) == 2:
            # add empty heading dimension if missing from target
            with torch.no_grad():
                target = torch.cat(
                    [
                        target[..., 0:2],
                        torch.zeros_like(target[..., 0:1]),
                    ],
                    dim=-1,
                )

        with torch.no_grad():
            preds = preds[..., -1, 0:3].clone().float().to(self.device).clone()
            if preds.ndim == 3:
                preds.unsqueeze(1)
            target = (
                target[..., -1, 0:3]
                .clone()
                .float()
                .unsqueeze(1)
                .to(self.device)
            )

            # project to the frame of the last observed position
            for idx in torch.unique(batch):
                _filter = batch == idx.item()
                preds[_filter] = (
                    project_poses(
                        preds[_filter],
                        anchor[idx.item()].to(self.device),
                        precision=torch.float32,
                        inverse=True,
                    )
                    .view(_filter.sum().item(), -1, 3)
                    .clone()
                )

                target[_filter] = (
                    project_poses(
                        target[_filter],
                        anchor[idx.item()].to(self.device),
                        precision=torch.float32,
                        inverse=True,
                    )
                    .view(_filter.sum().item(), -1, 3)
                    .clone()
                )
            displ = preds[..., 0:2] - target[..., 0:2]
            lng_thld = torch.ones_like(displ[..., 0]) * 2.0
            lat_thld = torch.ones_like(displ[..., 1]) * 2.0

            miss_flags = torch.add(
                displ[..., 0].abs() > lng_thld, displ[..., 1].abs() > lat_thld
            )
            self.total += torch.sum(torch.all(miss_flags, dim=1))
            self.count += target.size(0)

    def compute(self) -> torch.FloatTensor:
        return self.total.float() / (self.count + 1e-8) * 100
