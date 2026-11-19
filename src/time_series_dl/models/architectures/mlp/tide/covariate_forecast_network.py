from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....components.covariates import StaticCovariateEmbedding
from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors
from .dense_encoder_decoder import TiDEDenseEncoderDecoder
from .residual_block import TiDEResidualBlock

if TYPE_CHECKING:
    from .parameters import TiDEParameters


class TiDECovariateForecastNetwork(CovariateForecastNetwork):
    """
    Channel-independent TiDE with known covariates and static attributes, as in the paper.

    A residual block projects the known covariates of every step of the window and the horizon to
    ``covariate_projection_size`` features (the feature projection). The dense encoder reads each target channel's
    window together with the static attributes and the flattened projections of every step; the temporal decoder
    reads each horizon step's decoded vector with that step's projection. Categorical static features are embedded.
    Past-observed channels that are not targets are not read, as in the model without covariates.

    Parameters
    ----------
    parameters : TiDEParameters
        The model configuration.
    shape : CovariateForecastShape
        The dimensions of the forecast and of the covariates.
    """

    def __init__(self, parameters: TiDEParameters, shape: CovariateForecastShape) -> None:
        super().__init__()
        forecast = shape.forecast
        self.horizon: int = forecast.horizon
        projection_size: int = parameters.covariate_projection_size if shape.is_known_present else 0
        self.static_embedding: StaticCovariateEmbedding = StaticCovariateEmbedding(
            shape.static_channels, shape.static_cardinalities
        )
        self.feature_projection: TiDEResidualBlock | None = (
            TiDEResidualBlock(
                shape.known_channels,
                parameters.hidden_size,
                projection_size,
                parameters.dropout,
                parameters.is_layer_normalized,
            )
            if shape.is_known_present
            else None
        )
        shared_size: int = (
            self.static_embedding.output_size + (forecast.input_length + forecast.horizon) * projection_size
        )
        self.dense: TiDEDenseEncoderDecoder = TiDEDenseEncoderDecoder(
            parameters, forecast, shared_size, projection_size
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(forecast.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2)
        projected: Tensor = (
            covariates.known if self.feature_projection is None else self.feature_projection(covariates.known)
        )
        static: Tensor = self.static_embedding(covariates.static, covariates.static_categories)
        shared: Tensor = torch.cat([static, projected.flatten(start_dim=1)], dim=-1)
        return self.dense(series, shared, projected[:, -self.horizon :])
