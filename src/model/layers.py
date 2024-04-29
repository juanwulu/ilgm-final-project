# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Common neural network layers for the FAMOP motion prediction model."""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.pool import max_pool_x
from torch_geometric.typing import OptTensor

__all__ = [
    "MLP",
    "PolylineSubGraphLayer",
    "GlobalLayer",
    "compute_sinusoid_positional_encoding",
]


def compute_sinusoid_positional_encoding(
    pos: FloatTensor,
    dim: int,
    max_len: int = 10000,
) -> FloatTensor:
    """Compute the sinusoid positional encoding."""
    assert pos.ndim == 2 and pos.size(-1) == 2, (
        "Invalid position tensor. Expected a 2D tensor of shape (N, 2), "
        f"got {pos.shape}."
    )
    half_dim = dim // 2
    denominator = torch.exp(
        torch.arange(0, half_dim, device=pos.device, dtype=pos.dtype)
        * (-math.log(max_len) / half_dim)
    ).unsqueeze(0)
    x_emb = pos[:, 0:1] * 2 * math.pi / denominator
    y_emb = pos[:, 1:2] * 2 * math.pi / denominator
    x_emb = torch.stack([x_emb[:, 0::2].sin(), x_emb[:, 1::2].cos()], dim=-1)
    y_emb = torch.stack([y_emb[:, 0::2].sin(), y_emb[:, 1::2].cos()], dim=-1)
    pos_emb = torch.cat([x_emb.flatten(-2), y_emb.flatten(-2)], dim=-1)

    return pos_emb


class MLP(nn.Module):
    """Two-layer Multi-layer Perceptron with layer normalization."""

    # ----------- public attributes ----------- #
    in_features: int
    """int: Input feature dimensionality."""
    hidden_size: int
    """int: Hidden feature dimensionality."""
    out_feature: int
    """int: Output feature dimensionality."""

    # ----------- private modules ----------- #
    _hidden_linear: Linear
    """Linear: Hidden linear layer."""
    _layer_norm: Union[LayerNorm, nn.Identity]
    """Union[LayerNorm, nn.Identity]: Layer Normalization layer."""
    _output_linear: Optional[Linear]
    """Optional[Linear]: Output linear layer."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_feature: Optional[int] = None,
        for_graph: bool = False,
        has_norm: bool = True,
    ) -> None:
        super().__init__()

        # save arguments
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature

        # initialize network layers
        self._hidden_linear = Linear(
            in_channels=in_features,
            out_channels=hidden_size,
            bias=True,
        )
        if has_norm:
            self._layer_norm = LayerNorm(
                in_channels=(
                    hidden_size if self.out_feature is None else out_feature
                ),
                mode="graph" if for_graph else "node",
            )
        else:
            self._layer_norm = nn.Identity()

        if self.out_feature is not None:
            self._output_linear = Linear(
                in_channels=hidden_size, out_channels=out_feature, bias=True
            )
        else:
            self.register_module("_output_linear", None)

        self.reset_parameters()

    def forward(self, x: Tensor, batch: OptTensor = None) -> FloatTensor:
        output = self._hidden_linear.forward(x.float())
        output = F.silu(output)

        if self.out_feature is not None:
            output = self._output_linear.forward(x=output)

        if isinstance(self._layer_norm, LayerNorm):
            output = self._layer_norm.forward(x=output, batch=batch)
        else:
            output = self._layer_norm.forward(input=output)

        return output

    def reset_parameters(self) -> None:
        reset(self._hidden_linear)
        reset(self._layer_norm)
        if self.out_feature is not None:
            reset(self._output_linear)

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"in_features={self.in_features}",
                f"hidden_size={self.hidden_size}",
                f"out_feature={self.out_feature}",
            ]
        )
        return f"<MLP({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


class PolylineSubGraphLayer(nn.Module):
    """VectorNet-style polyline-level feature message passing network."""

    # ----------- public attributes ----------- #
    in_features: int
    """int: Input feature dimensionality of the polyline feature encoder."""
    hidden_size: int
    """int: Hidden feature dimensionality of the polyline feature encoder."""
    out_feature: int
    """int: Output feature dimensionality of the polyline feature encoder."""

    # ----------- private modules ----------- #
    _update_mlp: MLP
    """MLP: Node feature update MLP."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()

        # save arguments
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = in_features

        # initialize network layers
        self._update_mlp = MLP(
            in_features=in_features,
            hidden_size=hidden_size,
            out_feature=int(in_features // 2),
            for_graph=False,
            has_norm=True,
        )

    def forward(
        self, x: Tensor, clusters: Tensor, batch: Tensor
    ) -> FloatTensor:
        """Forward pass of the polyline-level feature message passing network.

        Args:
            x (Tensor): Input polyline segment features.
            clusters (Tensor): Cluster indices of each polyline segments.
            batch (Tensor): Batch indices of each polyline segments.

        Returns:
            FloatTensor: Updated polyline segment features.
        """
        x, batch, clusters = x.float(), batch.long(), clusters.long()

        # compute cluster-level features
        out = self._update_mlp.forward(x=x)
        # apply max pooling to compute cluster-level features
        aggr, _ = max_pool_x(cluster=clusters, x=out, batch=batch)
        # concatenate cluster-level features with polyline segment features
        out = torch.cat([out, aggr[clusters]], dim=-1)

        # apply L2 normalization to the updated polyline segment features
        out = out / torch.maximum(
            torch.norm(input=out, dim=-1, p=2, keepdim=True),
            torch.tensor(1e-12),
        )

        return out

    def reset_parameters(self) -> None:
        reset(self._update_mlp)

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"in_features={self.in_features}",
                f"hidden_size={self.hidden_size}",
            ]
        )
        return f"<PolylineSubGraphLayer({attr_str}) at {hex(id(self))}>"


class GlobalLayer(nn.Module):
    """Multi-head Attention Layer for Global Interaction modeling."""

    # ----------- public attributes ----------- #
    query_size: int
    """int: Query feature dimensionality."""
    key_size: int
    """int: Key feature dimensionality."""
    value_size: int
    """int: Value feature dimensionality."""
    num_heads: int
    """int: Number of attention heads."""

    # ----------- private modules ----------- #
    _mha_layer: nn.MultiheadAttention
    """nn.MultiheadAttention: Multi-head Attention layer."""
    _mha_layer_norm: LayerNorm
    """LayerNorm: Layer Normalization layer after the attention layer."""
    _ffn_layers: nn.Sequential
    """nn.Sequential: Feed-forward network layers."""
    _ffn_layer_norm: LayerNorm
    """LayerNorm: Output layer Normalization layer."""

    def __init__(
        self,
        query_features: int,
        key_features: int,
        value_features: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # save arguments
        self.query_size = query_features
        self.key_size = key_features
        self.value_size = value_features
        self.num_heads = num_heads

        # initialize attention network layers
        self._mha_layer = nn.MultiheadAttention(
            embed_dim=query_features,
            kdim=key_features,
            vdim=value_features,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self._mha_layer_norm = nn.LayerNorm(value_features)

        # initialize feed-forward network layers
        self._ffn_layers = nn.Sequential()
        self._ffn_layers.add_module(
            "Linear_1",
            nn.Linear(value_features, value_features * 4),
        )
        self._ffn_layers.add_module("Activation_1", nn.GELU())
        self._ffn_layers.add_module(
            "Linear_2", nn.Linear(value_features * 4, value_features)
        )
        self._ffn_layer_norm = nn.LayerNorm(value_features)

        self.reset_parameters()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pe: OptTensor,
        key_pe: OptTensor,
        *args,
        **kwargs,
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of the global interaction modeling layer.

        Args:
            query (Tensor): Query features.
            key (Tensor): Key features.
            value (Tensor): Value features.
            query_pe (OptTensor): Optional query position encoding.
            key_pe (OptTensor): Optional key position encoding.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Updated features and attention.
        """
        query, key, value = query.float(), key.float(), value.float()

        # forward pass of the attention layers
        mha_out, attn = self._mha_layer.forward(
            query=query + query_pe.float() if query_pe is not None else query,
            key=key + key_pe.float() if key_pe is not None else key,
            value=value,
            *args,
            **kwargs,
        )
        mha_out = self._mha_layer_norm.forward(mha_out + query)

        # forward pass of the feed-forward network layers
        ffn_out = self._ffn_layers.forward(mha_out)
        ffn_out = self._ffn_layer_norm.forward(ffn_out + mha_out)

        return ffn_out, attn

    def reset_parameters(self) -> None:
        reset(self._mha_layer)
        reset(self._mha_layer_norm)
        reset(self._ffn_layers)
        reset(self._ffn_layer_norm)
