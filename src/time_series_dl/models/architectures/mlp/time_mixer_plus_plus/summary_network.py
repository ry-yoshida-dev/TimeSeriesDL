from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import TimeMixerPlusPlusEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerPlusPlusParameters


class TimeMixerPlusPlusSummaryNetwork(SummaryNetwork):
    """
    The multi-scale encoder followed by a GELU, dropout and a linear map of the flattened tokens of every scale to the
    output vector.

    Parameters
    ----------
    parameters : TimeMixerPlusPlusParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.

    Raises
    ------
    ValueError
        If the input is too short for the down-samplings.
    """

    def __init__(self, parameters: TimeMixerPlusPlusParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: TimeMixerPlusPlusEncoder = TimeMixerPlusPlusEncoder(
            parameters, shape.input_length, shape.input_channels
        )
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(sum(self.encoder.lengths) * parameters.model_size, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: list[Tensor] = self.encoder(values)
        flattened: Tensor = torch.cat([sequence.flatten(start_dim=1) for sequence in tokens], dim=1)
        outputs: Tensor = self.projection(self.dropout(self.activation(flattened)))
        return outputs
