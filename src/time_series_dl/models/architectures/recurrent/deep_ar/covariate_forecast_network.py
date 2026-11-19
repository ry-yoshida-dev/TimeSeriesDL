from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....components.covariates import StaticCovariateEmbedding
from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors
from .recurrence import DeepARRecurrence

if TYPE_CHECKING:
    from .parameters import DeepARParameters


class DeepARCovariateForecastNetwork(CovariateForecastNetwork):
    """
    DeepAR with covariates, as in the paper: every step's recurrent input is concatenated with that step's known
    covariates and with the static features of the series, whose categorical codes (such as an item id) are embedded.

    Parameters
    ----------
    parameters : DeepARParameters
        The model configuration.
    shape : CovariateForecastShape
        The dimensions of the forecast and of the covariates.
    """

    def __init__(self, parameters: DeepARParameters, shape: CovariateForecastShape) -> None:
        super().__init__()
        forecast = shape.forecast
        self.input_length: int = forecast.input_length
        self.static_embedding: StaticCovariateEmbedding = StaticCovariateEmbedding(
            shape.static_channels, shape.static_cardinalities
        )
        step_feature_size: int = shape.known_channels + self.static_embedding.output_size
        self.recurrence: DeepARRecurrence = DeepARRecurrence(
            parameters,
            forecast.input_channels + step_feature_size,
            forecast.output_channels,
            step_feature_size,
            forecast.num_parameters,
            forecast.location_index,
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(forecast.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        static: Tensor = self.static_embedding(covariates.static, covariates.static_categories).unsqueeze(1)
        features: Tensor = torch.cat([covariates.known, static.expand(-1, covariates.known.shape[1], -1)], dim=-1)
        last_targets: Tensor = past.index_select(2, self.target_channels)[:, -1:, :]
        return self.recurrence(
            torch.cat([past, features[:, : self.input_length]], dim=-1),
            last_targets,
            features[:, self.input_length :],
            future_targets,
        )
