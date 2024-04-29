# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Base Data Class."""
from __future__ import annotations, print_function

import abc
from typing import Any, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch_geometric.data import Data

from src.utils.logging import get_pylogger

# Constants
LOGGER = get_pylogger(__name__)


class BaseData(abc.ABC, Data):
    """Base data class representing the traffic environment."""

    # map features
    x_map: FloatTensor
    """FloatTensor: Map polyline features of shape `[n_line * n_points, M]`."""
    map_clusters: LongTensor
    """LongTensor: The cluster label based on polyline index of map points."""
    map_centroid: FloatTensor
    """FloatTensor: The centroid of the polylines with shape `[n_line, 2]`."""

    # agent features
    x_motion: FloatTensor
    """FloatTensor: The motion state features."""
    motion_cluster: LongTensor
    """LongTensor: The cluster label based on agent index of motion states."""
    motion_timestamps: LongTensor
    """LongTensor: The timestamps of each motion states."""
    track_ids: LongTensor
    """LongTensor: The ground-truth agent id of each track."""
    track_pos: FloatTensor
    """FloatTensor: The last observed agent position of each track."""

    # target subgraph
    y_motion: FloatTensor
    """The ground-truth future motion state features of the agents."""
    y_valid: BoolTensor
    """The ground-truth future motion state validity of the agents."""
    y_cluster: LongTensor
    """The ground-truth future motion state cluster based on agent indexes."""

    # system info
    anchor: FloatTensor
    """FloatTensor: A 2D pose the anchor of this sample with shape `[3,]`."""
    tracks_to_predict: LongTensor
    """LongTensor: The agent ids of the tracks to predict."""
    num_agents: LongTensor
    """LongTensor: The number of agents in the scene in this sample."""

    # metadata
    sample_idx: LongTensor
    """LongTensor: The sample index of this sample in the source dataset."""
    follow_batch: Tuple[str] = (
        "x_map",
        "x_motion",
        "track_ids",
        "y_motion",
        "tracks_to_predict",
    )
    """Tuple[str]: The tuple of attributes to follow batch."""
    num_data: LongTensor
    """LongTensor: The number of samples in the dataset."""

    def is_valid(self) -> bool:
        """Sanity check for the data model.

        Returns:
            bool: True if the data model is valid, False otherwise.
        """
        map_valid = self._is_map_valid()
        track_valid = self._is_track_valid()
        return map_valid and track_valid

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "map_clusters":
            return len(self.map_clusters.unique())
        if key == "motion_cluster":
            return self.track_ids.size(0)
        elif key in ["y_cluster", "track_ids", "tracks_to_predict"]:
            return self.num_agents.item()
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ["anchor", "num_agents"]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    @abc.abstractmethod
    def map_feature_dims(self) -> int:
        """int: The dimension of the map features."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def motion_feature_dims(self) -> int:
        """int: The dimension of motion state features."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def target_dims(self) -> int:
        """int: The dimension of the target motion state features."""
        raise NotImplementedError

    def _is_map_valid(self) -> bool:
        return (
            # map point data sanity checks
            self.x_map.size(0) > 0
            and self.x_map.size(1) == self.map_feature_dims
            and self.map_clusters.size(0) == self.x_map.size(0)
            and self.map_clusters.min().item() >= 0
        )

    def _is_track_valid(self) -> bool:
        return (
            # motion state data sanity checks
            self.x_motion.size(0) > 0
            and self.x_motion.size(1) == self.motion_feature_dims
            and self.motion_cluster.size(0) == self.x_motion.size(0)
            and self.motion_cluster.min().item() >= 0
            and self.motion_cluster.max().item() == self.track_ids.size(0) - 1
            and self.motion_timestamps.size(0) == self.x_motion.size(0)
            and self.track_ids.size(0) == self.motion_cluster.unique().size(0)
            and self.track_ids.min().item() >= 0
            # target data sanity checks
            and (
                torch.all(
                    torch.isin(self.tracks_to_predict, self.y_cluster)
                ).item()
                if len(self.y_cluster) > 0
                else True
            )
        )
