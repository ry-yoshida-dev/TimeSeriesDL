from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.layers import TemporalPooling
from ....interfaces import SummaryNetwork, SummaryShape

if TYPE_CHECKING:
    from .parameters import TSMixerParameters


class TSMixerSummaryNetwork(SummaryNetwork):
    """
    Mixer blocks, then temporal pooling and a linear output layer.

    Parameters
    ----------
    parameters : TSMixerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: TSMixerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.mixer: nn.Sequential = parameters.build_mixer(shape.input_length, shape.input_channels)
        self.pooling: TemporalPooling = parameters.pooling
        self.projection: nn.Linear = nn.Linear(
            parameters.pooling.output_size(shape.input_length, shape.input_channels), shape.output_size
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        mixed: Tensor = self.mixer(values)
        outputs: Tensor = self.projection(self.pooling.apply(mixed))
        return outputs
