from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .model import ChronosModel

if TYPE_CHECKING:
    from .parameters import ChronosParameters


class ChronosForecastNetwork(GenerativeForecastNetwork):
    """
    Chronos applied to each target channel as a separate univariate series; the other channels are not read.

    Forecasts are sample paths drawn token by token, and fine-tuning minimizes the token cross-entropy of the true
    future values with teacher forcing, as in the paper's training.

    Parameters
    ----------
    parameters : ChronosParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast; ``num_parameters`` is the number of sample paths of ``forward``.
    """

    def __init__(self, parameters: ChronosParameters, shape: ForecastShape) -> None:
        super().__init__(shape.num_parameters)
        self.model: ChronosModel = ChronosModel(parameters)
        self.horizon: int = shape.horizon
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def compute_loss(self, past: Tensor, future_targets: Tensor) -> Tensor:
        return self.model.compute_loss(self._to_series(past), future_targets.transpose(1, 2).flatten(end_dim=1))

    @override
    def sample(self, past: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        batch: int = past.shape[0]
        samples: Tensor = self.model.sample(self._to_series(past), self.horizon, num_samples, generator)
        return samples.reshape(batch, -1, num_samples, self.horizon).permute(0, 2, 3, 1)

    def _to_series(self, past: Tensor) -> Tensor:
        return past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
