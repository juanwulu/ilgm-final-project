# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Helper functions for dataset."""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.transforms import BaseTransform

from .base_data import BaseData


# --------- Geometry Operators ---------- #
def project_poses(
    poses: torch.Tensor,
    anchor: torch.Tensor,
    precision: Optional[torch.dtype] = None,
    rotate: bool = False,
    inverse: bool = False,
) -> torch.Tensor:
    """Affine transform input `poses` to new system centered about `anchor`.

    Args:
        poses (torch.Tensor): local states, of shape `[*, 3]`.
        anchor (torch.Tensor): global anchor, of shape `[*, 3]`.
        precision (Optional[torch.dtype], optional): precision of the output
            tensor. If `None`, use `poses` precision. Defaults to `None`.
        rotate (bool, optional): if apply rotation. Defaults to `False`.
        inverse (bool, optional): if apply inverse transform. If `True`,
            project the target-centric state back to the global frame.
            Defaults to `False`.

    Returns:
        torch.Tensor: global states, of shape `[*, 3]`.
    """
    assert poses.size(-1) == 3, ValueError(
        "Invalid input `poses`. Expect last dimension to be 3."
    )
    assert anchor.size(-1) == 3, ValueError(
        "Invalid input `anchor`. Expect last dimension to be 3."
    )

    if anchor.ndim == 1:
        # unsqueeze batch dimension
        anchor = anchor.unsqueeze(0)
    if poses.ndim == anchor.ndim + 1:
        # expand anchor to match poses
        anchor = anchor.unsqueeze(-2)

    assert poses.ndim == anchor.ndim, "Unmatched input shapes."

    if precision is None:
        precision = poses.dtype
    output = torch.zeros_like(poses, device=poses.device, dtype=precision)

    # unpack local states and global anchor
    x, y, theta = poses.unbind(dim=-1)
    x0, y0, theta0 = anchor.unbind(dim=-1)

    if rotate:
        # compute rotation
        cosine = torch.cos(theta0)
        sine = torch.sin(theta0)

        if inverse:
            x_new = cosine * x - sine * y + x0
            y_new = sine * x + cosine * y + y0
            theta_new = (theta + theta0) % (2 * torch.pi)
        else:
            x_new = cosine * (x - x0) + sine * (y - y0)
            y_new = -sine * (x - x0) + cosine * (y - y0)
            theta_new = (theta - theta0) % (2 * torch.pi)
    else:
        if inverse:
            x_new = x + x0
            y_new = y + y0
            theta_new = theta
        else:
            x_new = x - x0
            y_new = y - y0
            theta_new = theta

    return torch.stack((x_new, y_new, theta_new), dim=-1, out=output)


def project_velocities(
    vels: torch.Tensor,
    anchor: torch.Tensor,
    precision: Optional[torch.dtype] = None,
    rotate: bool = True,
    inverse: bool = False,
) -> torch.Tensor:
    """Project velocities from global frame to target-centric frame.

    Args:
        vels (torch.Tensor): global velocities of shape `[*, 2]` or `[2, ]`.
        anchor (torch.Tensor): global anchor of shape `[*, 3]` or `[3, ]`.
        precision (Optional[torch.dtype], optional): precision of the output
            tensor. If `None`, use `poses` precision. Defaults to `None`.
        rotate (bool, optional): if apply rotation. Defaults to `True`.
        inverse (bool, optional): if apply inverse transform. If `True`,
            project the target-centric velocity back to the global frame.
            Defaults to `False`.

    Returns:
        torch.Tensor: target-centric velocities of shape `[*, 2]`.
    """
    assert vels.size(-1) == 2, "Invalid input `vels`. Expect last dim to be 2."
    assert (
        anchor.size(-1) == 3
    ), "Invalid input `anchor`. Expect last dim to be 3."

    if anchor.ndim == 1:
        # unsqueeze batch dimension
        anchor = anchor.unsqueeze(0)
    if vels.ndim == anchor.ndim + 1:
        # expand anchor to match vels
        anchor = anchor.unsqueeze(-2)

    assert vels.ndim == anchor.ndim, "Unmatched input shapes."

    if precision is None:
        precision = vels.dtype
    output = torch.zeros_like(vels, device=vels.device, dtype=precision)

    # compute rotation
    vx, vy = vels.unbind(dim=-1)
    _, _, theta0 = anchor.unbind(dim=-1)
    cosine = torch.cos(theta0)
    sine = torch.sin(theta0)

    if rotate:
        if inverse:
            raise NotImplementedError("Inverse transform not implemented.")
        else:
            vx_new = cosine * vx + sine * vy
            vy_new = -sine * vx + cosine * vy
    else:
        if inverse:
            raise NotImplementedError("Inverse transform not implemented.")
        else:
            vx_new = vx
            vy_new = vy

    return torch.stack((vx_new, vy_new), dim=-1, out=output)


class TargetCentricTransform(BaseTransform):
    """Project global coordinates into a target-centric local frame system."""

    rotate: bool
    """bool: If apply rotation."""

    def __init__(self, rotate: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rotate = rotate

    def __call__(self, data: BaseData) -> BaseData:
        assert isinstance(data, BaseData), f"Unsupported type: {type(data)}"
        with torch.no_grad():
            anchor = data.anchor
            assert anchor.shape == (1, 3) or anchor.shape == (3,)
            data.x_map[:, 0:2] = project_poses(
                poses=torch.hstack(
                    [
                        data.x_map[:, 0:2],
                        torch.zeros_like(data.x_map[:, 0:1]),
                    ]
                ),
                anchor=anchor,
                precision=torch.float32,
                rotate=self.rotate,
            )[:, 0:2]
            data.map_centroid[:, 0:2] = project_poses(
                poses=torch.hstack(
                    [
                        data.map_centroid[:, 0:2],
                        torch.zeros_like(data.map_centroid[:, 0:1]),
                    ]
                ),
                anchor=anchor,
                precision=torch.float32,
                rotate=self.rotate,
            )[:, 0:2]
            if data.x_motion.size(-1) == 3:
                data.x_motion[:, 0:2] = project_poses(
                    poses=torch.hstack(
                        [
                            data.x_motion[:, 0:2],
                            torch.zeros_like(data.x_motion[:, 0:1]),
                        ]
                    ),
                    anchor=anchor,
                    precision=torch.float32,
                    rotate=self.rotate,
                )[:, 0:2]
            else:
                data.x_motion[:, 0:3] = project_poses(
                    poses=data.x_motion[:, 0:3],
                    anchor=anchor,
                    precision=torch.float32,
                    rotate=self.rotate,
                )
            if data.x_motion.size(-1) > 4:
                data.x_motion[:, 3:5] = project_velocities(
                    vels=data.x_motion[:, 3:5],
                    anchor=anchor,
                    precision=torch.float32,
                    rotate=self.rotate,
                )
            data.track_pos[:, 0:2] = project_poses(
                poses=torch.hstack(
                    [
                        data.track_pos[:, 0:2],
                        torch.zeros_like(data.track_pos[:, 0:1]),
                    ]
                ),
                anchor=anchor,
                precision=torch.float32,
                rotate=self.rotate,
            )[:, 0:2]
            data.y_motion[..., 0:2] = project_poses(
                poses=torch.hstack(
                    [
                        data.y_motion[..., 0:2],
                        torch.zeros_like(data.y_motion[..., 0:1]),
                    ]
                ),
                anchor=anchor,
                precision=torch.float32,
                rotate=self.rotate,
            )[:, 0:2]

        return data

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__}(rotate={self.rotate}) "
            f"at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        return str(self)


class TargetReshapeTransform(BaseTransform):
    """Reshape target motion states."""

    prediction_horizon: int
    """int: The prediction horizon in number of waypoints."""

    def __init__(self, prediction_horizon: int = 30) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon

    def __call__(self, data: BaseData) -> BaseData:
        assert isinstance(data, BaseData), f"Unsupported type: {type(data)}."
        agent_ids: torch.LongTensor = data.y_cluster.unique()

        size = (
            agent_ids.size(0),
            self.prediction_horizon,
            data.target_dims,
        )
        new_y_motion = torch.zeros(
            size=size, device=data.y_motion.device, dtype=torch.float32
        )
        new_y_cluster = torch.zeros(
            size=size[0:1], device=data.y_cluster.device, dtype=torch.long
        )
        new_y_valid = torch.zeros(
            size=size[0:2], device=data.y_valid.device, dtype=torch.bool
        )

        for idx, cluster in enumerate(agent_ids):
            mask = data.y_cluster == cluster
            num_states = torch.nonzero(mask).size(0)
            new_y_motion[idx, 0:num_states] = data.y_motion[mask]
            new_y_cluster[idx] = cluster
            new_y_valid[idx, 0:num_states] = True

        data.y_motion = new_y_motion
        data.y_cluster = new_y_cluster
        data.y_valid = new_y_valid

        return data

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)
