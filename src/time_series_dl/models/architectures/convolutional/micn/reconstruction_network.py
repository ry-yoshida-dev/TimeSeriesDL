from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....interfaces import ReconstructionNetwork, SequenceShape
from .multiscale_decomposition import MICNMultiScaleDecomposition
from .seasonal_prediction import MICNSeasonalPrediction

if TYPE_CHECKING:
    from .parameters import MICNParameters


class MICNReconstructionNetwork(ReconstructionNetwork):
    """
    MICN for imputation: the seasonal part of the sequence goes through the seasonal prediction block, and its trend
    is added back.

    Parameters
    ----------
    parameters : MICNParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: MICNParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.decomposition: MICNMultiScaleDecomposition = MICNMultiScaleDecomposition(parameters.decomposition_sizes)
        self.seasonal_prediction: MICNSeasonalPrediction = MICNSeasonalPrediction(
            parameters, shape.channels, shape.channels, shape.length
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        seasonal: Tensor
        trend: Tensor
        seasonal, trend = self.decomposition(values)
        reconstructed: Tensor = self.seasonal_prediction(seasonal) + trend
        return reconstructed
