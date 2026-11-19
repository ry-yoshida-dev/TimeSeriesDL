import math

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SummaryShape
from time_series_dl.models.architectures.transformer.etsformer import (
    ETSformerDampingLayer,
    ETSformerExponentialSmoothing,
    ETSformerFrequencyAttention,
    ETSformerParameters,
)

MODELS: list[ETSformerParameters] = [
    ETSformerParameters(model_size=8, num_heads=2, num_layers=2, feedforward_size=16, top_k=2, dropout=0.1),
    ETSformerParameters(model_size=6, num_heads=3, num_layers=1, feedforward_size=8, top_k=1, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ETSformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("input_length", [2, 19])
@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: ETSformerParameters, input_length: int) -> None:
    check_summary_network(model, SummaryShape(input_length=input_length, input_channels=3, output_size=4))


@pytest.mark.parametrize("is_auxiliary", [False, True])
def test_exponential_smoothing_matches_recurrence(is_auxiliary: bool) -> None:
    _ = torch.manual_seed(0)
    smoothing = ETSformerExponentialSmoothing(num_heads=3, head_size=2, dropout=0.0)
    values: Tensor = torch.randn(2, 7, 3, 2)
    auxiliary: Tensor = torch.randn(2, 7, 3, 2)
    with torch.no_grad():
        outputs: Tensor = smoothing(values, auxiliary if is_auxiliary else None)
        weight: Tensor = smoothing.smoothing_weight.view(1, 3, 1)
        state: Tensor = smoothing.initial_state.unsqueeze(0).expand(2, -1, -1)
        expected: list[Tensor] = []
        for step in range(7):
            previous: Tensor = state + auxiliary[:, step] if is_auxiliary else state
            state = weight * previous + (1.0 - weight) * values[:, step]
            expected.append(state)
    assert torch.allclose(outputs, torch.stack(expected, dim=1), atol=1e-5)


def test_damping_sums_powers_of_the_factor() -> None:
    _ = torch.manual_seed(0)
    damping = ETSformerDampingLayer(horizon=4, num_heads=2, dropout=0.0)
    growth: Tensor = torch.randn(3, 1, 6)
    with torch.no_grad():
        outputs: Tensor = damping(growth)
        factor: Tensor = damping.damping_factor.repeat_interleave(3)
    for step in range(4):
        scale: Tensor = sum((factor**power for power in range(1, step + 2)), torch.zeros(6))
        assert torch.allclose(outputs[:, step], growth[:, 0] * scale, atol=1e-6)


def test_frequency_attention_extrapolates_a_sinusoid() -> None:
    length: int = 24
    horizon: int = 10
    times: Tensor = torch.arange(length + horizon, dtype=torch.float32)
    wave: Tensor = 1.5 * torch.cos(2.0 * math.pi * 3.0 * times / length + 0.4)
    attention = ETSformerFrequencyAttention(top_k=1, horizon=horizon)
    outputs: Tensor = attention(wave[:length].view(1, length, 1))
    assert outputs.shape == (1, length + horizon, 1)
    assert torch.allclose(outputs[0, :, 0], wave, atol=1e-4)


def test_frequency_attention_without_components_is_zero() -> None:
    attention = ETSformerFrequencyAttention(top_k=3, horizon=2)
    assert bool((attention(torch.randn(2, 2, 4)) == 0).all())


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        ETSformerParameters(num_layers=0)
    with pytest.raises(ValueError, match="top_k"):
        ETSformerParameters(top_k=0)
    with pytest.raises(ValueError, match="num_heads"):
        ETSformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        ETSformerParameters(dropout=1.0)
