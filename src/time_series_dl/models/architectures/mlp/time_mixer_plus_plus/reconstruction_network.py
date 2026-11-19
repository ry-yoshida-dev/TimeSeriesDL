from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import TimeMixerPlusPlusEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerPlusPlusParameters


class TimeMixerPlusPlusReconstructionNetwork(ReconstructionNetwork):
    """
    The multi-scale encoder and one linear map per scale along time back to the length of the sequence; the maps are
    summed and each time step's tokens projected to its values.

    Parameters
    ----------
    parameters : TimeMixerPlusPlusParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If the sequences are too short for the down-samplings.
    """

    def __init__(self, parameters: TimeMixerPlusPlusParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: TimeMixerPlusPlusEncoder = TimeMixerPlusPlusEncoder(parameters, shape.length, shape.channels)
        self.predictors: nn.ModuleList = nn.ModuleList(
            nn.Linear(length, shape.length) for length in self.encoder.lengths
        )
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.channels)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: list[Tensor] = self.encoder(values)
        mapped: list[Tensor] = [
            predictor(sequence.transpose(1, 2)).transpose(1, 2)
            for predictor, sequence in zip(self.predictors, tokens, strict=True)
        ]
        reconstructed: Tensor = self.projection(torch.stack(mapped, dim=0).sum(dim=0))
        return reconstructed
