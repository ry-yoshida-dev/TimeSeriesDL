from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....interfaces import ReconstructionNetwork, SequenceShape
from .frequency_interpolation import FITSFrequencyInterpolation

if TYPE_CHECKING:
    from .parameters import FITSParameters


class FITSReconstructionNetwork(ReconstructionNetwork):
    """
    Low-pass filter and complex linear map of the spectrum of each channel back onto the window.

    Parameters
    ----------
    parameters : FITSParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: FITSParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.interpolation: FITSFrequencyInterpolation = FITSFrequencyInterpolation(
            shape.length,
            shape.length,
            parameters.compute_cutoff_frequency(shape.length),
            shape.channels,
            parameters.is_individual,
            1,
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        return self.interpolation(values).squeeze(-1)
