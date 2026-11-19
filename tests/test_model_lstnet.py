import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SummaryShape
from time_series_dl.models.architectures.recurrent.lstnet import (
    LSTNetForecastNetwork,
    LSTNetParameters,
    LSTNetSequenceEncoder,
)

MODELS: list[LSTNetParameters] = [
    LSTNetParameters(
        convolution_channels=6,
        convolution_size=1,
        recurrent_size=5,
        skip_recurrent_size=3,
        skip_period=2,
        highway_window=3,
    ),
    LSTNetParameters(
        convolution_channels=4,
        convolution_size=2,
        recurrent_size=3,
        skip_recurrent_size=2,
        skip_period=0,
        highway_window=0,
        dropout=0.0,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: LSTNetParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: LSTNetParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=11, input_channels=3, output_size=4))


def test_skip_component_reads_each_phase_one_period_apart() -> None:
    _ = torch.manual_seed(0)
    parameters = LSTNetParameters(convolution_channels=4, convolution_size=2, recurrent_size=3, skip_period=3)
    encoder = LSTNetSequenceEncoder(parameters, 12, 2).eval()
    values: Tensor = torch.randn(2, 12, 2)
    with torch.no_grad():
        features: Tensor = encoder(values).squeeze(1)
        convolved: Tensor = torch.relu(encoder.convolution(values.transpose(1, 2))).transpose(1, 2)
        assert encoder.skip_recurrent is not None
        # The convolved length 11 holds three whole periods of 3, the last 9 steps.
        phases: list[Tensor] = [encoder.skip_recurrent(convolved[:, 2 + phase :: 3])[1][0] for phase in range(3)]
        expected: Tensor = torch.cat([encoder.recurrent(convolved)[1][0], *phases], dim=-1)
    assert torch.allclose(features, expected, atol=1e-6)


def test_highway_adds_a_shared_linear_map_of_each_target() -> None:
    _ = torch.manual_seed(0)
    shape = ForecastShape(input_length=10, input_channels=3, horizon=4, target_channels=(2, 0), num_parameters=2)
    network = LSTNetForecastNetwork(MODELS[0], shape).eval()
    past: Tensor = torch.randn(2, 10, 3)
    highway = network.highway
    assert highway is not None
    with torch.no_grad():
        outputs: Tensor = network(past)
        neural: Tensor = network.projection(network.encoder(past).squeeze(1)).reshape(2, 4, 2, 2)
        expected: Tensor = torch.stack([highway(past[:, -3:, channel]) for channel in (2, 0)], dim=1)
    assert torch.allclose(outputs - neural, expected.reshape(2, 2, 4, 2).transpose(1, 2), atol=1e-6)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="convolution_size"):
        LSTNetParameters(convolution_size=0)
    with pytest.raises(ValueError, match="skip_period"):
        LSTNetParameters(skip_period=-1)
    with pytest.raises(ValueError, match="highway_window"):
        LSTNetParameters(highway_window=-1)
    with pytest.raises(ValueError, match="dropout"):
        LSTNetParameters(dropout=1.0)
    shape = ForecastShape(input_length=20, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="convolution_size"):
        LSTNetParameters(convolution_size=21, skip_period=0, highway_window=0).build_forecast_network(shape)
    with pytest.raises(ValueError, match="skip_period"):
        LSTNetParameters(highway_window=4).build_forecast_network(shape)
    with pytest.raises(ValueError, match="highway_window"):
        LSTNetParameters(skip_period=0).build_forecast_network(shape)
