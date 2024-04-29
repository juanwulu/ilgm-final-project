# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Encoder module for the motion prediction model."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.nn.pool import max_pool_x

from ..data.base_data import BaseData
from .layers import (
    MLP,
    GlobalLayer,
    PolylineSubGraphLayer,
    compute_sinusoid_positional_encoding,
)
from .typing import StateDict

__all__ = ["PolylineSubGraph", "Encoder"]


class PolylineSubGraph(nn.Module):
    """VectorNet-style polyline subgraph module."""

    # ----------- public attributes ----------- #
    in_features: int
    """int: Input feature dimensionality."""
    out_feature: int
    """int: Output feature dimensionality."""
    num_layers: int
    """int: Number of subgraph layers in the subgraph module."""

    # ----------- private module ------------ #
    _pre_mlp: Union[MLP, nn.Identity]
    """Union[MLP, nn.Identity]: MLP for pre-processing input features."""
    _layers: nn.ModuleList
    """nn.ModuleList: List of subgraph layers."""
    _output_mlp: Union[MLP, nn.Identity]
    """Union[MLP, nn.Identity]: MLP for post-processing output features."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_feature: Optional[int] = None,
        num_layers: int = 3,
        has_pre_mlp: bool = True,
    ) -> None:
        """Construct a `PolylineSubGraph` object."""
        super().__init__()

        # save arguments
        self.in_features = in_features
        self.num_layers = num_layers
        self.out_feature = out_feature or hidden_size

        if has_pre_mlp:
            self._pre_mlp = MLP(
                in_features=in_features,
                hidden_size=hidden_size,
                out_feature=hidden_size,
                for_graph=False,
                has_norm=True,
            )
            in_features = hidden_size
        else:
            self._pre_mlp = nn.Identity()

        self._layers = nn.ModuleList()
        for _ in range(1, num_layers + 1):
            self._layers.append(
                PolylineSubGraphLayer(
                    in_features=in_features,
                    hidden_size=hidden_size,
                )
            )

        if out_feature is not None:
            self._output_mlp = MLP(
                in_features=in_features,
                hidden_size=hidden_size,
                out_feature=out_feature,
                for_graph=False,
                has_norm=True,
            )
        else:
            self._output_mlp = nn.Identity()

        self.reset_parameters()

    def forward(
        self, x: Tensor, clusters: Tensor, batch: Tensor
    ) -> Tuple[FloatTensor, LongTensor]:
        x, batch, clusters = x.float(), batch.long(), clusters.long()

        # forward pass the pre-MLP
        x = self._pre_mlp.forward(x)

        # forward pass the subgraph layers
        out = x
        for layer in self._layers:
            assert isinstance(layer, PolylineSubGraphLayer)
            out = layer.forward(x=out, clusters=clusters, batch=batch)

        # aggregate the output
        out, batch = max_pool_x(cluster=clusters, x=out, batch=batch)

        # forward pass the output MLP
        out = self._output_mlp.forward(out)

        # apply L2 normalization to the output
        out = out / torch.maximum(
            torch.linalg.norm(out, ord=2, dim=-1, keepdim=True),
            torch.tensor(1e-12),
        )

        return out, batch

    def reset_parameters(self) -> None:
        reset(self._pre_mlp)
        reset(self._layers)

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"in_features={self.in_features}",
                f"out_feature={self.out_feature}",
                f"num_layers={self.num_layers}",
            ]
        )
        return f"<PolylineSubGraph({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


class Encoder(nn.Module):
    """Encoder module wrapping MapNet and MotionNet for feature encoding."""

    # ----------- public attributes ----------- #
    horizon: int | float
    """int: Observation horizon in number of waypoints."""

    # ----------- private module ------------ #
    _mapnet: PolylineSubGraph
    """PolylineSubGraph: MapNet module for map polyline feature encoding."""
    _motionnet: PolylineSubGraph
    """PolylineSubGraph: MotionNet module for motion state feature encoding."""
    _context_module: nn.ModuleList
    """nn.ModuleList: Global attention module for context encoding."""
    _interaction_module: nn.ModuleList
    """nn.ModuleList: Global attention module for interaction encoding."""

    def __init__(
        self,
        map_in_features: int,
        map_hidden_size: int,
        map_num_layers: int,
        motion_in_features: int,
        motion_hidden_size: int,
        motion_num_layers: int,
        map_out_size: Optional[int] = None,
        motion_out_size: Optional[int] = None,
        dropout: float = 0.1,
        num_heads: int = 8,
        horizon: int = 10,
    ) -> None:
        super().__init__()

        # save arguments
        self.horizon = horizon

        # initialize map encoder
        self._mapnet = PolylineSubGraph(
            in_features=map_in_features,
            hidden_size=map_hidden_size,
            out_feature=map_out_size,
            num_layers=map_num_layers,
            has_pre_mlp=True,
        )

        # initialize motion encoder
        self._motionnet = PolylineSubGraph(
            in_features=motion_in_features,
            hidden_size=motion_hidden_size,
            out_feature=motion_out_size,
            num_layers=motion_num_layers,
            has_pre_mlp=True,
        )

        # initialize global attention modules
        self._context_module = nn.ModuleList(
            [
                GlobalLayer(
                    query_features=self._motionnet.out_feature,
                    key_features=self._motionnet.out_feature,
                    value_features=self._motionnet.out_feature,
                    dropout=dropout,
                    num_heads=num_heads,
                )
                for _ in range(3)
            ]
        )
        self._interaction_module = nn.ModuleList(
            [
                GlobalLayer(
                    query_features=self._motionnet.out_feature,
                    key_features=self._motionnet.out_feature,
                    value_features=self._motionnet.out_feature,
                    dropout=dropout,
                    num_heads=num_heads,
                )
                for _ in range(3)
            ]
        )

        self.reset_parameters()

    @property
    def out_feature(self) -> int:
        """int: Output feature dimension."""
        return self._motionnet.out_feature

    def forward(self, data: BaseData, need_weights: bool = False) -> StateDict:
        with torch.set_grad_enabled(self.training):
            # encode map features
            map_feats, map_clusters = data.x_map, data.map_clusters
            map_batch = data.get("x_map_batch", torch.zeros_like(data.x_map))
            map_feats, map_batch = self._mapnet.forward(
                x=map_feats, clusters=map_clusters, batch=map_batch
            )
            map_cntrs = data.map_centroid.float()
            assert map_batch.size(0) == map_feats.size(0) == map_cntrs.size(0)

            # encode motion features
            motion_feats, motion_clusters = data.x_motion, data.motion_cluster
            motion_batch = data.get(
                "x_motion_batch", torch.zeros_like(data.x_motion)
            )
            track_feats, track_batch = self._motionnet.forward(
                x=motion_feats, clusters=motion_clusters, batch=motion_batch
            )
            track_ids = data.track_ids.long()
            assert (
                track_batch.size(0) == track_feats.size(0) == track_ids.size(0)
            )

            # only keep the motion features of the tracks to predict
            with torch.no_grad():
                tar_filter = torch.isin(data.track_ids, data.tracks_to_predict)
                tar_filter = tar_filter.view(-1)

                assert data.motion_timestamps.unique().size(0) == self.horizon
                current_state = data.x_motion[
                    torch.isin(
                        data.track_ids[data.motion_cluster],
                        data.tracks_to_predict,
                    ).view(-1)
                    & (data.motion_timestamps == data.motion_timestamps.max())
                ]

            # compute and add positional encoding
            map_pe = compute_sinusoid_positional_encoding(
                pos=map_cntrs, dim=map_feats.size(-1)
            )
            track_pe = compute_sinusoid_positional_encoding(
                pos=data.track_pos[:, 0:2], dim=track_feats.size(-1)
            )

            # encode context features for expected locations
            context_batch = torch.cat(
                [track_batch[tar_filter], map_batch], dim=0
            )
            context_feats = torch.cat(
                [track_feats[tar_filter], map_feats], dim=0
            )
            context_pe = torch.cat([track_pe[tar_filter], map_pe], dim=0)
            attn_mask = context_batch.view(-1, 1) != context_batch.view(1, -1)
            for lyr in self._context_module:
                if isinstance(lyr, GlobalLayer):
                    context_feats, _ = lyr.forward(
                        query=context_feats,
                        key=context_feats,
                        value=context_feats,
                        query_pe=context_pe,
                        key_pe=context_pe,
                        need_weights=need_weights,
                        attn_mask=attn_mask,
                    )
            context_feats = context_feats[: tar_filter.sum().item()]

            # encode interaction features for uncertainty
            interaction_batch = torch.cat(
                [track_batch[tar_filter], track_batch[~tar_filter]], dim=0
            )
            interaction_feats = torch.cat(
                [track_feats[tar_filter], track_feats[~tar_filter]], dim=0
            )
            interaction_pe = torch.cat(
                [track_pe[tar_filter], track_pe[~tar_filter]], dim=0
            )
            attn_mask = (
                interaction_batch[:, None] != interaction_batch[None, :]
            )
            for lyr in self._interaction_module:
                if isinstance(lyr, GlobalLayer):
                    interaction_feats, _ = lyr.forward(
                        query=interaction_feats,
                        key=interaction_feats,
                        value=interaction_feats,
                        query_pe=interaction_pe,
                        key_pe=interaction_pe,
                        need_weights=need_weights,
                        attn_mask=attn_mask,
                    )
            interaction_feats = interaction_feats[: tar_filter.sum().item()]

            output = {
                "target_feats": track_feats[tar_filter],
                "context_feats": context_feats + interaction_feats,
                "interaction_feats": interaction_feats,
                "current_state": current_state,
            }

        return output

    def reset_parameters(self) -> None:
        reset(self._mapnet)
        reset(self._motionnet)
        reset(self._context_module)
        reset(self._interaction_module)

    @property
    def motionnet(self) -> PolylineSubGraph:
        """PolylineSubGraph: MotionNet module."""
        return self._motionnet

    @property
    def mapnet(self) -> PolylineSubGraph:
        """PolylineSubGraph: MapNet module."""
        return self._mapnet

    @property
    def context_module(self) -> nn.ModuleList:
        """nn.ModuleList: Global attention module for context encoding."""
        return self._context_module

    @property
    def interaction_module(self) -> nn.ModuleList:
        """nn.ModuleList: Global attention module for interaction encoding."""
        return self._interaction_module

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"mapnet={self._mapnet}",
                f"motionnet={self._motionnet}",
            ]
        )
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)
