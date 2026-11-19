import math

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.autoformer import AutoformerParameters
from time_series_dl.models.components.attention import AutoCorrelationAttentionMechanism

MODELS: list[AutoformerParameters] = [
    AutoformerParameters(model_size=8, num_heads=2, feedforward_size=16, moving_average_size=5, label_length=2),
    AutoformerParameters(
        model_size=6,
        num_heads=3,
        num_encoder_layers=1,
        num_decoder_layers=2,
        feedforward_size=8,
        moving_average_size=25,
        factor=1,
        activation=Activation.RELU,
        label_length=3,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: AutoformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: AutoformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: AutoformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_auto_correlation_selects_period_delays() -> None:
    period: int = 8
    steps: Tensor = torch.arange(32, dtype=torch.float32)
    series: Tensor = torch.sin(2 * math.pi * steps / period).view(1, 1, -1, 1).expand(2, 2, -1, 3)
    mechanism = AutoCorrelationAttentionMechanism(1)
    delays: Tensor
    weights: Tensor
    delays, weights = mechanism.compute_delays(series, series)
    assert delays.shape == (2, int(math.log(32)))
    assert bool((delays % period == 0).all())
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.allclose(mechanism(series, series, series), series, atol=1e-5)


def test_auto_correlation_aligns_keys_to_queries() -> None:
    mechanism = AutoCorrelationAttentionMechanism(2)
    queries: Tensor = torch.randn(2, 1, 10, 4)
    for key_length in (6, 14):
        keys: Tensor = torch.randn(2, 1, key_length, 4)
        assert mechanism(queries, keys, keys).shape == queries.shape


def test_untrained_trend_starts_from_label_trend_and_mean() -> None:
    shape = ForecastShape(
        input_length=12,
        input_channels=3,
        horizon=4,
        target_channels=(2, 0),
        num_parameters=2,
        level_parameters=(0,),
    )
    network = AutoformerParameters(
        model_size=8, num_heads=2, feedforward_size=16, moving_average_size=5, label_length=6
    ).build_forecast_network(shape)
    for parameter in network.parameters():
        _ = parameter.data.zero_()
    _ = network.eval()
    past: Tensor = torch.randn(2, 12, 3)
    outputs: Tensor = network(past)
    target_mean: Tensor = past[:, :, [2, 0]].mean(dim=1, keepdim=True).expand(-1, 4, -1)
    assert torch.allclose(outputs[..., 0], target_mean, atol=1e-6)
    assert torch.allclose(outputs[..., 1], torch.zeros(2, 4, 2))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="moving_average_size"):
        AutoformerParameters(moving_average_size=24)
    with pytest.raises(ValueError, match="factor"):
        AutoformerParameters(factor=0)
    with pytest.raises(ValueError, match="factor"):
        AutoCorrelationAttentionMechanism(0)
    shape = ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="label_length"):
        AutoformerParameters(model_size=8, num_heads=2, label_length=9).build_forecast_network(shape)
