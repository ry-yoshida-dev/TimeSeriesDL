from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .image_decomposition import TimeMixerPlusPlusImageDecomposition
from .scale_mixing import TimeMixerPlusPlusScaleMixing
from .time_imaging import TimeMixerPlusPlusTimeImaging

if TYPE_CHECKING:
    from .parameters import TimeMixerPlusPlusParameters


class TimeMixerPlusPlusMixerBlock(nn.Module):
    """
    Mixer block of TimeMixer++: multi-resolution time imaging, time image decomposition, multi-scale mixing, and
    multi-resolution mixing of the periods weighted by the softmax of their amplitudes, with a residual connection.

    Parameters
    ----------
    parameters : TimeMixerPlusPlusParameters
        The model configuration.
    lengths : tuple[int, ...]
        Number of time steps of every scale, from the finest to the coarsest.
    """

    def __init__(self, parameters: TimeMixerPlusPlusParameters, lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.lengths: tuple[int, ...] = lengths
        self.imaging: TimeMixerPlusPlusTimeImaging = TimeMixerPlusPlusTimeImaging(top_k=parameters.top_k)
        self.decomposition: TimeMixerPlusPlusImageDecomposition = TimeMixerPlusPlusImageDecomposition(
            parameters.model_size, parameters.num_heads, parameters.dropout
        )
        self.mixing: TimeMixerPlusPlusScaleMixing = TimeMixerPlusPlusScaleMixing(parameters.model_size, len(lengths))

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Mix the tokens of every scale.

        Parameters
        ----------
        tokens : list[Tensor]
            Tokens from the finest to the coarsest scale, each with shape (batch, length of the scale, model_size).

        Returns
        -------
        list[Tensor]
            Mixed tokens with the shapes of ``tokens``.
        """
        periods: list[int]
        weights: Tensor
        periods, weights = self.imaging.detect(tokens[-1], self.lengths[0])
        per_period: list[list[Tensor]] = []
        for period in periods:
            seasonal: list[Tensor] = []
            trend: list[Tensor] = []
            for sequence in tokens:
                parts: tuple[Tensor, Tensor] = self.decomposition(self.imaging.to_image(sequence, period))
                seasonal.append(parts[0])
                trend.append(parts[1])
            mixed: list[Tensor] = self.mixing(seasonal, trend)
            per_period.append(
                [self.imaging.to_sequence(image, length) for image, length in zip(mixed, self.lengths, strict=True)]
            )
        outputs: list[Tensor] = []
        for scale, sequence in enumerate(tokens):
            stacked: Tensor = torch.stack([sequences[scale] for sequences in per_period], dim=-1)
            outputs.append(sequence + (stacked * weights[:, None, None, :]).sum(dim=-1))
        return outputs

    def __call__(self, tokens: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(tokens))
