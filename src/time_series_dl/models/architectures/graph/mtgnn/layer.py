from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.graph import MixHopPropagation
from .dilated_inception import MTGNNDilatedInception

if TYPE_CHECKING:
    from .parameters import MTGNNParameters


class MTGNNLayer(nn.Module):
    """
    One layer of MTGNN: a gated dilated inception convolution along time, a skip connection to the output module,
    mix-hop propagation along the learned edges and against them, a residual connection and layer normalization.

    Parameters
    ----------
    parameters : MTGNNParameters
        The model configuration.
    num_nodes : int
        Number of nodes.
    dilation : int
        Dilation of the temporal convolutions.
    output_length : int
        Number of steps left after the temporal convolutions, which the skip convolution reduces to one.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, parameters: MTGNNParameters, num_nodes: int, dilation: int, output_length: int) -> None:
        if num_nodes <= 0 or output_length <= 0:
            raise ValueError(f"num_nodes and output_length must be positive, got {num_nodes} and {output_length}")
        super().__init__()
        self.filter: MTGNNDilatedInception = MTGNNDilatedInception(
            parameters.residual_channels, parameters.convolution_channels, dilation
        )
        self.gate: MTGNNDilatedInception = MTGNNDilatedInception(
            parameters.residual_channels, parameters.convolution_channels, dilation
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.skip: nn.Conv2d = nn.Conv2d(parameters.convolution_channels, parameters.skip_channels, (1, output_length))
        self.forward_propagation: MixHopPropagation = MixHopPropagation(
            parameters.convolution_channels,
            parameters.residual_channels,
            parameters.propagation_depth,
            parameters.retain_ratio,
        )
        self.backward_propagation: MixHopPropagation = MixHopPropagation(
            parameters.convolution_channels,
            parameters.residual_channels,
            parameters.propagation_depth,
            parameters.retain_ratio,
        )
        self.norm: nn.LayerNorm = nn.LayerNorm([parameters.residual_channels, num_nodes, output_length])

    def forward(self, features: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the layer.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, residual_channels, nodes, length).
        adjacency : Tensor
            Learned adjacency with shape (nodes, nodes).

        Returns
        -------
        tuple[Tensor, Tensor]
            Features with shape (batch, residual_channels, nodes, output_length), and the skip contribution with
            shape (batch, skip_channels, nodes, 1).
        """
        gated: Tensor = torch.tanh(self.filter(features)) * torch.sigmoid(self.gate(features))
        gated = self.dropout(gated)
        skip: Tensor = self.skip(gated)
        propagated: Tensor = self.forward_propagation(gated, adjacency) + self.backward_propagation(
            gated, adjacency.transpose(0, 1)
        )
        normalized: Tensor = self.norm(propagated + features[..., -propagated.shape[-1] :])
        return normalized, skip

    def __call__(self, features: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(features, adjacency))
