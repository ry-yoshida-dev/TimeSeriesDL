from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape

if TYPE_CHECKING:
    from .parameters import TSMixerParameters


class TSMixerReconstructionNetwork(ReconstructionNetwork):
    """
    Mixer blocks, then a linear map along time of each channel back onto the window.

    Parameters
    ----------
    parameters : TSMixerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: TSMixerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.mixer: nn.Sequential = parameters.build_mixer(shape.length, shape.channels)
        self.projection: nn.Linear = nn.Linear(shape.length, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        mixed: Tensor = self.mixer(values)
        projected: Tensor = self.projection(mixed.transpose(1, 2))
        return projected.transpose(1, 2)
