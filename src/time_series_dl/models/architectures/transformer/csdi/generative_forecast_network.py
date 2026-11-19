from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .conditional_imputer import CSDIConditionalImputer

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDIForecastNetwork(GenerativeForecastNetwork):
    """
    CSDI forecasting as imputation: the past window and the horizon of every input channel form one
    (channels, input_length + horizon) array, the past is conditioned on, and the horizon of the target channels is
    imputed. The horizon of channels that are not targets is neither conditioned on nor trained on.

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast; ``num_parameters`` is the number of sample paths of ``forward``.
    """

    def __init__(self, parameters: CSDIParameters, shape: ForecastShape) -> None:
        super().__init__(shape.num_parameters)
        self.input_length: int = shape.input_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.imputer: CSDIConditionalImputer = parameters.build_conditional_imputer(shape.input_channels)
        total: int = shape.input_length + shape.horizon
        conditional_mask: Tensor = torch.zeros(shape.input_channels, total)
        conditional_mask[:, : shape.input_length] = 1.0
        observed_mask: Tensor = conditional_mask.clone()
        observed_mask[list(shape.target_channels), shape.input_length :] = 1.0
        self.target_channels: Tensor
        self.conditional_mask: Tensor
        self.observed_mask: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)
        self.register_buffer("conditional_mask", conditional_mask, persistent=False)
        self.register_buffer("observed_mask", observed_mask, persistent=False)

    @override
    def compute_loss(self, past: Tensor, future_targets: Tensor) -> Tensor:
        expected: tuple[int, int, int] = (past.shape[0], self.horizon, self.output_channels)
        if tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}, got {tuple(future_targets.shape)}")
        values: Tensor = self._assemble(past, future_targets)
        batch: int = past.shape[0]
        return self.imputer.compute_loss(
            values,
            self.observed_mask.to(values.dtype).expand(batch, -1, -1),
            self.conditional_mask.to(values.dtype).expand(batch, -1, -1),
        )

    @override
    def sample(self, past: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        values: Tensor = self._assemble(past, past.new_zeros(past.shape[0], self.horizon, self.output_channels))
        samples: Tensor = self.imputer.impute(
            values, self.conditional_mask.to(values.dtype).expand(past.shape[0], -1, -1), num_samples, generator
        )
        horizon: Tensor = samples[..., self.input_length :].index_select(2, self.target_channels)
        return horizon.transpose(2, 3)

    def _assemble(self, past: Tensor, future_targets: Tensor) -> Tensor:
        future: Tensor = past.new_zeros(past.shape[0], self.horizon, past.shape[2]).index_copy(
            2, self.target_channels, future_targets
        )
        return torch.cat([past, future], dim=1).transpose(1, 2)
