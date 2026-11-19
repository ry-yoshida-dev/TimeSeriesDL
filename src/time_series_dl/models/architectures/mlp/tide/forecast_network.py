from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .dense_encoder_decoder import TiDEDenseEncoderDecoder

if TYPE_CHECKING:
    from .parameters import TiDEParameters


class TiDEForecastNetwork(ForecastNetwork):
    """
    Channel-independent TiDE without covariates.

    The forecast of each target channel reads only that channel's past values: its window is encoded and decoded by
    dense residual blocks into one vector per horizon step, each vector is mapped to the distribution parameters by
    the temporal decoder, and a linear map of the window to the horizon is added. All weights are shared by the
    channels. The temporal decoder is never layer-normalized, since its outputs are the distribution parameters.

    Parameters
    ----------
    parameters : TiDEParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TiDEParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.dense: TiDEDenseEncoderDecoder = TiDEDenseEncoderDecoder(parameters, shape, 0, 0)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2)
        return self.dense(series, past.new_zeros(past.shape[0], 0), past.new_zeros(past.shape[0], self.horizon, 0))
