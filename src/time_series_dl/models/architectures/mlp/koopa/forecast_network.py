from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.layers import Activation, MultilayerPerceptron
from ....interfaces import ForecastNetwork, ForecastShape
from .fourier_filter import KoopaFourierFilter
from .time_invariant_predictor import KoopaTimeInvariantPredictor
from .time_variant_predictor import KoopaTimeVariantPredictor

if TYPE_CHECKING:
    from .parameters import KoopaParameters


class KoopaForecastNetwork(ForecastNetwork):
    """
    Stacked Koopa blocks: each block splits its input by the Fourier filter, forecasts the time-invariant part with
    the global operator and the time-variant part with each window's operator, and passes on the input minus the
    reconstructed time-variant part; the block forecasts are summed and mapped to the distribution parameters.

    Segments are as long as the horizon, so the window must be longer than the horizon.

    Parameters
    ----------
    parameters : KoopaParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If the input is not longer than the horizon.
    """

    def __init__(self, parameters: KoopaParameters, shape: ForecastShape) -> None:
        if shape.input_length <= shape.horizon:
            raise ValueError(
                f"Koopa needs an input longer than the horizon, got {shape.input_length} and {shape.horizon}"
            )
        super().__init__()
        hidden_sizes: list[int] = [parameters.hidden_size] * (parameters.num_layers - 1)
        segment_size: int = shape.horizon * shape.input_channels
        self.filter: KoopaFourierFilter = KoopaFourierFilter(shape.input_length, parameters.invariant_fraction)
        invariant_encoder: MultilayerPerceptron = MultilayerPerceptron(
            shape.input_length, hidden_sizes, parameters.dynamic_size, Activation.RELU, parameters.dropout
        )
        invariant_decoder: MultilayerPerceptron = MultilayerPerceptron(
            parameters.dynamic_size, hidden_sizes, shape.horizon, Activation.RELU, parameters.dropout
        )
        variant_encoder: MultilayerPerceptron = MultilayerPerceptron(
            segment_size, hidden_sizes, parameters.dynamic_size, Activation.TANH, parameters.dropout
        )
        variant_decoder: MultilayerPerceptron = MultilayerPerceptron(
            parameters.dynamic_size, hidden_sizes, segment_size, Activation.TANH, parameters.dropout
        )
        self.invariant_predictors: nn.ModuleList = nn.ModuleList(
            KoopaTimeInvariantPredictor(invariant_encoder, invariant_decoder, parameters.dynamic_size)
            for _ in range(parameters.num_blocks)
        )
        self.variant_predictors: nn.ModuleList = nn.ModuleList(
            KoopaTimeVariantPredictor(
                variant_encoder, variant_decoder, shape.horizon, shape.input_length, shape.horizon
            )
            for _ in range(parameters.num_blocks)
        )
        self.parameter_layer: nn.Linear = nn.Linear(1, shape.num_parameters)
        with torch.no_grad():
            _ = self.parameter_layer.weight.fill_(1.0)
            _ = self.parameter_layer.bias.zero_()
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        residual: Tensor = past
        block_forecasts: list[Tensor] = []
        for invariant_predictor, variant_predictor in zip(
            self.invariant_predictors, self.variant_predictors, strict=True
        ):
            variant: Tensor
            invariant: Tensor
            variant, invariant = self.filter(residual)
            backcast: Tensor
            variant_forecast: Tensor
            backcast, variant_forecast = variant_predictor(variant)
            residual = residual - backcast
            block_forecasts.append(invariant_predictor(invariant) + variant_forecast)
        forecast: Tensor = torch.stack(block_forecasts).sum(dim=0)
        targets: Tensor = forecast.index_select(2, self.target_channels)
        outputs: Tensor = self.parameter_layer(targets.unsqueeze(-1))
        return outputs
