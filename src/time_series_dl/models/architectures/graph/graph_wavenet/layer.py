from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.graph import DiffusionConvolution

if TYPE_CHECKING:
    from .parameters import GraphWaveNetParameters


class GraphWaveNetLayer(nn.Module):
    """
    One spatial-temporal layer of Graph WaveNet: a gated dilated convolution along time, a skip connection, a
    diffusion graph convolution, a residual connection and batch normalization.

    Parameters
    ----------
    parameters : GraphWaveNetParameters
        The model configuration.
    dilation : int
        Dilation of the temporal convolutions.
    is_propagating : bool
        Whether the layer computes the features of a next layer; the last layer only contributes its skip
        connection, so its graph convolution, which would receive no gradient, is left out.

    Raises
    ------
    ValueError
        If dilation is not positive.
    """

    def __init__(self, parameters: GraphWaveNetParameters, dilation: int, is_propagating: bool) -> None:
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        super().__init__()
        kernel: tuple[int, int] = (1, parameters.kernel_size)
        self.filter: nn.Conv2d = nn.Conv2d(
            parameters.residual_channels, parameters.dilation_channels, kernel, dilation=(1, dilation)
        )
        self.gate: nn.Conv2d = nn.Conv2d(
            parameters.residual_channels, parameters.dilation_channels, kernel, dilation=(1, dilation)
        )
        self.skip: nn.Conv2d = nn.Conv2d(parameters.dilation_channels, parameters.skip_channels, 1)
        self.graph_convolution: DiffusionConvolution | None = (
            DiffusionConvolution(
                parameters.dilation_channels,
                parameters.residual_channels,
                1,
                parameters.diffusion_order,
                parameters.dropout,
            )
            if is_propagating
            else None
        )
        self.norm: nn.BatchNorm2d | None = nn.BatchNorm2d(parameters.residual_channels) if is_propagating else None

    def forward(self, features: Tensor, transition: Tensor) -> tuple[Tensor | None, Tensor]:
        """
        Apply the layer.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, residual_channels, nodes, length).
        transition : Tensor
            Transition matrix with shape (nodes, nodes); entry (i, j) weighs the message node i receives from node j.

        Returns
        -------
        tuple[Tensor | None, Tensor]
            Features with shape (batch, residual_channels, nodes, output_length), or None if the layer does not
            propagate, and the skip contribution with shape (batch, skip_channels, nodes, output_length), where
            output_length is length minus (kernel_size - 1) * dilation.
        """
        gated: Tensor = torch.tanh(self.filter(features)) * torch.sigmoid(self.gate(features))
        skip: Tensor = self.skip(gated)
        if self.graph_convolution is None or self.norm is None:
            return None, skip
        convolved: Tensor = self.graph_convolution(gated, (transition,))
        normalized: Tensor = self.norm(convolved + features[..., -convolved.shape[-1] :])
        return normalized, skip

    def __call__(self, features: Tensor, transition: Tensor) -> tuple[Tensor | None, Tensor]:
        return cast(tuple[Tensor | None, Tensor], super().__call__(features, transition))
