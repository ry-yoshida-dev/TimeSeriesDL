from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .model import TTMModel

if TYPE_CHECKING:
    from .parameters import TTMParameters


class TTMForecastNetwork(ForecastNetwork):
    """
    TinyTimeMixer applied to each target channel as a separate univariate series, as the published channel-independent
    checkpoints are; the other channels are not read. The forecast is the first ``horizon`` values of the model's
    prediction.

    Parameters
    ----------
    parameters : TTMParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If ``num_parameters`` is not 1 or the horizon exceeds the model's ``prediction_length``.
    """

    def __init__(self, parameters: TTMParameters, shape: ForecastShape) -> None:
        if shape.num_parameters != 1:
            raise ValueError(
                f"TinyTimeMixer forecasts points only; num_parameters must be 1, got {shape.num_parameters}"
            )
        if shape.horizon > parameters.prediction_length:
            raise ValueError(
                f"horizon {shape.horizon} exceeds the prediction_length {parameters.prediction_length}; "
                + "choose a variant with a longer prediction length."
            )
        super().__init__()
        self.model: TTMModel = TTMModel(parameters)
        self.horizon: int = shape.horizon
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
        forecast: Tensor = self.model(series)[:, : self.horizon]
        return forecast.reshape(batch, -1, self.horizon).transpose(1, 2).unsqueeze(-1)
