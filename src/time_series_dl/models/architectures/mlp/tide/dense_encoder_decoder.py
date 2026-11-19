from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....interfaces import ForecastShape
from .residual_block import TiDEResidualBlock

if TYPE_CHECKING:
    from .parameters import TiDEParameters


class TiDEDenseEncoderDecoder(nn.Module):
    """
    The forecast of TiDE for each target channel: the dense encoder reads the channel's window with the features
    shared by every channel (static attributes and projected covariates), the dense decoder maps the encoding to one
    vector per horizon step, the temporal decoder maps each vector with that step's projected covariates to the
    distribution parameters, and a linear map of the window to the horizon is added.

    Parameters
    ----------
    parameters : TiDEParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    shared_size : int
        Size of the features shared by the channels that the dense encoder reads next to each window.
    step_feature_size : int
        Size of the features of each horizon step that the temporal decoder reads.
    """

    def __init__(
        self, parameters: TiDEParameters, shape: ForecastShape, shared_size: int, step_feature_size: int
    ) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.decoder_output_size: int = parameters.decoder_output_size
        self.encoder: nn.Sequential = parameters.build_dense_stack(
            shape.input_length + shared_size, parameters.hidden_size, parameters.num_encoder_layers
        )
        self.decoder: nn.Sequential = parameters.build_dense_stack(
            parameters.hidden_size, shape.horizon * parameters.decoder_output_size, parameters.num_decoder_layers
        )
        self.temporal_decoder: TiDEResidualBlock = TiDEResidualBlock(
            parameters.decoder_output_size + step_feature_size,
            parameters.temporal_decoder_hidden_size,
            shape.num_parameters,
            parameters.dropout,
            is_layer_normalized=False,
        )
        self.residual: nn.Linear = nn.Linear(shape.input_length, shape.horizon * shape.num_parameters)

    def forward(self, series: Tensor, shared: Tensor, step_features: Tensor) -> Tensor:
        """
        Forecast every target channel.

        Parameters
        ----------
        series : Tensor
            Windows of the target channels with shape (batch, output_channels, input_length).
        shared : Tensor
            Features shared by the channels with shape (batch, shared_size).
        step_features : Tensor
            Features of each horizon step with shape (batch, horizon, step_feature_size).

        Returns
        -------
        Tensor
            Unconstrained distribution parameters with shape (batch, horizon, output_channels, num_parameters).
        """
        batch_size: int = series.shape[0]
        channels: int = series.shape[1]
        encoding: Tensor = self.encoder(torch.cat([series, shared.unsqueeze(1).expand(-1, channels, -1)], dim=-1))
        decoded: Tensor = self.decoder(encoding)
        steps: Tensor = decoded.reshape(batch_size, channels, self.horizon, self.decoder_output_size)
        features: Tensor = step_features.unsqueeze(1).expand(-1, channels, -1, -1)
        temporal: Tensor = self.temporal_decoder(torch.cat([steps, features], dim=-1))
        residual: Tensor = self.residual(series).reshape(batch_size, channels, self.horizon, self.num_parameters)
        return (temporal + residual).transpose(1, 2)

    def __call__(self, series: Tensor, shared: Tensor, step_features: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(series, shared, step_features))
