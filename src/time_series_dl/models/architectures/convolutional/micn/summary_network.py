from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .multiscale_decomposition import MICNMultiScaleDecomposition
from .seasonal_prediction import MICNSeasonalPrediction

if TYPE_CHECKING:
    from .parameters import MICNParameters


class MICNSummaryNetwork(SummaryNetwork):
    """
    MICN for classification and regression: the seasonal prediction of every step plus its trend, followed by GELU,
    dropout and a linear map of all flattened steps to the output vector.

    Parameters
    ----------
    parameters : MICNParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: MICNParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.decomposition: MICNMultiScaleDecomposition = MICNMultiScaleDecomposition(parameters.decomposition_sizes)
        self.seasonal_prediction: MICNSeasonalPrediction = MICNSeasonalPrediction(
            parameters, shape.input_channels, shape.input_channels, shape.input_length
        )
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(shape.input_length * shape.input_channels, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        seasonal: Tensor
        trend: Tensor
        seasonal, trend = self.decomposition(values)
        features: Tensor = self.dropout(self.activation(self.seasonal_prediction(seasonal) + trend))
        outputs: Tensor = self.projection(features.flatten(start_dim=1))
        return outputs
