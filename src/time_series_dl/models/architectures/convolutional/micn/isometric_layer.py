from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.layers import SeriesDecomposition

if TYPE_CHECKING:
    from .parameters import MICNParameters

_FEEDFORWARD_RATIO: int = 4


class MICNIsometricLayer(nn.Module):
    """
    Multi-scale isometric convolution (MIC) layer of MICN.

    For each scale, the seasonal part of the input is downsampled by a strided convolution (local features), mixed by
    an isometric convolution whose kernel spans the whole downsampled sequence, padded at the front with zeros
    (global correlations), and upsampled back by a transposed convolution, each with tanh and dropout, and with
    residual connections and layer normalization. The scales are merged by a 2D convolution across scales, followed
    by a pointwise feed-forward network, as in the Time-Series-Library.

    Parameters
    ----------
    parameters : MICNParameters
        The model configuration.
    length : int
        Number of time steps of the sequence.
    """

    def __init__(self, parameters: MICNParameters, length: int) -> None:
        super().__init__()
        model_size: int = parameters.model_size
        kernel_sizes: tuple[int, ...] = parameters.kernel_sizes
        self.decompositions: nn.ModuleList = nn.ModuleList(
            SeriesDecomposition(size) for size in parameters.decomposition_sizes
        )
        self.downsampling: nn.ModuleList = nn.ModuleList(
            nn.Conv1d(model_size, model_size, size, stride=size, padding=size // 2) for size in kernel_sizes
        )
        self.isometric: nn.ModuleList = nn.ModuleList(
            nn.Conv1d(model_size, model_size, (length + 2 * (size // 2) - size) // size + 1) for size in kernel_sizes
        )
        self.upsampling: nn.ModuleList = nn.ModuleList(
            nn.ConvTranspose1d(model_size, model_size, size, stride=size) for size in kernel_sizes
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.merge: nn.Conv2d = nn.Conv2d(model_size, model_size, (len(kernel_sizes), 1))
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Conv1d(model_size, model_size * _FEEDFORWARD_RATIO, 1),
            nn.Conv1d(model_size * _FEEDFORWARD_RATIO, model_size, 1),
        )
        self.merge_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.output_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.activation: nn.Tanh = nn.Tanh()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Tokens with the shape of ``tokens``.
        """
        scales: list[Tensor] = [
            self._convolve_scale(self.decompositions[scale](tokens)[0], scale)
            for scale in range(len(self.decompositions))
        ]
        stacked: Tensor = torch.stack(scales, dim=1).permute(0, 3, 1, 2)
        merged: Tensor = self.merge(stacked).squeeze(-2).transpose(1, 2)
        hidden: Tensor = self.feedforward(self.merge_norm(merged).transpose(1, 2)).transpose(1, 2)
        outputs: Tensor = self.output_norm(merged + hidden)
        return outputs

    def _convolve_scale(self, seasonal: Tensor, scale: int) -> Tensor:
        length: int = seasonal.shape[1]
        local: Tensor = self.dropout(self.activation(self.downsampling[scale](seasonal.transpose(1, 2))))
        padded: Tensor = torch.cat([local.new_zeros(*local.shape[:2], local.shape[2] - 1), local], dim=-1)
        correlated: Tensor = self.dropout(self.activation(self.isometric[scale](padded)))
        combined: Tensor = self.norm((correlated + local).transpose(1, 2)).transpose(1, 2)
        upsampled: Tensor = self.dropout(self.activation(self.upsampling[scale](combined)))[:, :, :length]
        restored: Tensor = self.norm(upsampled.transpose(1, 2) + seasonal)
        return restored

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
