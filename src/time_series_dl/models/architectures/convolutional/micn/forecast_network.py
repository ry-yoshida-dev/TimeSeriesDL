from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .multiscale_decomposition import MICNMultiScaleDecomposition
from .seasonal_prediction import MICNSeasonalPrediction

if TYPE_CHECKING:
    from .parameters import MICNParameters


class MICNForecastNetwork(ForecastNetwork):
    """
    MICN forecaster: the past is decomposed; the trend of each target channel is extended over the horizon by a
    linear map along time, and the seasonal part, followed by zeros for the horizon, goes through the seasonal
    prediction block. The forecast is the seasonal output of the horizon plus, for the parameters in the scale of the
    values, the extended trend.

    Parameters
    ----------
    parameters : MICNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: MICNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.decomposition: MICNMultiScaleDecomposition = MICNMultiScaleDecomposition(parameters.decomposition_sizes)
        self.trend_regression: nn.Linear = nn.Linear(shape.input_length, shape.horizon)
        _ = nn.init.constant_(self.trend_regression.weight, 1.0 / shape.horizon)
        self.seasonal_prediction: MICNSeasonalPrediction = MICNSeasonalPrediction(
            parameters,
            shape.input_channels,
            shape.output_channels * shape.num_parameters,
            shape.input_length + shape.horizon,
        )
        is_level: Tensor = torch.zeros(shape.num_parameters)
        is_level[list(shape.level_parameters)] = 1.0
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        seasonal: Tensor
        trend: Tensor
        seasonal, trend = self.decomposition(past)
        target_trend: Tensor = trend.index_select(2, self.target_channels)
        future_trend: Tensor = self.trend_regression(target_trend.transpose(1, 2)).transpose(1, 2)
        placeholder: Tensor = past.new_zeros(past.shape[0], self.horizon, past.shape[2])
        seasonal_output: Tensor = self.seasonal_prediction(torch.cat([seasonal, placeholder], dim=1))
        outputs: Tensor = seasonal_output[:, -self.horizon :].reshape(
            -1, self.horizon, self.output_channels, self.num_parameters
        )
        return outputs + future_trend.unsqueeze(-1) * self.is_level
