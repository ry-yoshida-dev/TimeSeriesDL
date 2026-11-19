from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.layers import SeriesDecomposition
from ....interfaces import ForecastNetwork, ForecastShape
from ..xlstm import XLSTMBlockStack

if TYPE_CHECKING:
    from .parameters import XLSTMTimeParameters


class XLSTMTimeForecastNetwork(ForecastNetwork):
    """
    Decomposition, linear forecasts and an xLSTM stack over the channels.

    The window is split into a moving-average trend and the seasonal remainder; a linear map along time, initialized
    to the average, maps each part of each channel to the horizon, and the two are added. Each channel's preliminary
    forecast is embedded linearly and batch normalized over channels, the xLSTM stack reads the channels as a sequence
    (in input order), and a linear layer maps each channel's features to its distribution parameters over the horizon.

    Parameters
    ----------
    parameters : XLSTMTimeParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: XLSTMTimeParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.decomposition: SeriesDecomposition = SeriesDecomposition(parameters.moving_average_size)
        self.seasonal_linear: nn.Linear = nn.Linear(shape.input_length, shape.horizon)
        self.trend_linear: nn.Linear = nn.Linear(shape.input_length, shape.horizon)
        with torch.no_grad():
            _ = self.seasonal_linear.weight.fill_(1.0 / shape.input_length)
            _ = self.trend_linear.weight.fill_(1.0 / shape.input_length)
        self.embedding: nn.Linear = nn.Linear(shape.horizon, parameters.model_size)
        self.batch_normalization: nn.BatchNorm1d = nn.BatchNorm1d(shape.input_channels)
        self.stack: XLSTMBlockStack = parameters.stack.build(parameters.model_size)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        seasonal, trend = self.decomposition(past)
        preliminary: Tensor = self.seasonal_linear(seasonal.transpose(1, 2)) + self.trend_linear(trend.transpose(1, 2))
        tokens: Tensor = self.batch_normalization(self.embedding(preliminary))
        features: Tensor = self.stack(tokens).index_select(1, self.target_channels)
        outputs: Tensor = self.projection(features).reshape(*features.shape[:-1], self.horizon, self.num_parameters)
        return outputs.transpose(1, 2)
