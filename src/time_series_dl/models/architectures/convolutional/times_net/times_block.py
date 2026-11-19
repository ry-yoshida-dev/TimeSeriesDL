from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .inception_block import InceptionBlock
from .period_detector import PeriodDetector

if TYPE_CHECKING:
    from .parameters import TimesNetParameters


class TimesBlock(nn.Module):
    """
    Two-dimensional variation modeling at each dominant period, aggregated by amplitude, with a residual connection.

    For each detected period, the sequence is zero-padded at the end to a multiple of the period and folded into a
    grid of (cycles, period) on which inception convolutions capture intra- and inter-period variation. The results
    are weighted by the softmax of the period amplitudes, added to the input and layer-normalized.

    Parameters
    ----------
    parameters : TimesNetParameters
        The model configuration.
    """

    def __init__(self, parameters: TimesNetParameters) -> None:
        super().__init__()
        self.detector: PeriodDetector = PeriodDetector(top_k=parameters.top_k)
        self.convolution: nn.Sequential = nn.Sequential(
            InceptionBlock(parameters.model_size, parameters.feedforward_size, parameters.num_kernels),
            nn.GELU(),
            InceptionBlock(parameters.feedforward_size, parameters.model_size, parameters.num_kernels),
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Features with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Features of the same shape.
        """
        batch: int = values.shape[0]
        length: int = values.shape[1]
        size: int = values.shape[2]
        periods: list[int]
        amplitudes: Tensor
        periods, amplitudes = self.detector.detect(values)
        variations: list[Tensor] = []
        for period in periods:
            padded_length: int = -(-length // period) * period
            padded: Tensor = functional.pad(values, (0, 0, 0, padded_length - length))
            grid: Tensor = padded.reshape(batch, padded_length // period, period, size).permute(0, 3, 1, 2)
            convolved: Tensor = self.convolution(grid)
            variations.append(convolved.permute(0, 2, 3, 1).reshape(batch, padded_length, size)[:, :length])
        weights: Tensor = amplitudes.softmax(dim=1)
        mixed: Tensor = (torch.stack(variations, dim=-1) * weights[:, None, None, :]).sum(dim=-1)
        normalized: Tensor = self.norm(mixed + values)
        return normalized

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
