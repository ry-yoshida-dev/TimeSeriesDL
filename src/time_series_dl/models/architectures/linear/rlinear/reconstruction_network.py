from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.layers import TemporalLinear
from ....components.normalization import InstanceStatistics, ReversibleInstanceNormalization
from ....interfaces import ReconstructionNetwork, SequenceShape

if TYPE_CHECKING:
    from .parameters import RLinearParameters


class RLinearReconstructionNetwork(ReconstructionNetwork):
    """
    Reversible instance normalization, dropout, and a linear map along time back onto the window.

    Parameters
    ----------
    parameters : RLinearParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: RLinearParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.normalization: ReversibleInstanceNormalization = ReversibleInstanceNormalization(
            shape.channels, parameters.is_affine
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.linear: TemporalLinear = TemporalLinear(
            shape.length, shape.length, shape.channels, parameters.is_individual
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        statistics: InstanceStatistics = self.normalization.compute_statistics(values)
        normalized: Tensor = self.dropout(self.normalization.normalize(values, statistics))
        return self.normalization.denormalize(self.linear(normalized), statistics)
