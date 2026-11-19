import pytest
from forecast_shape_cases import FORECAST_SHAPE_CASES
from graph_network_checks import check_channel_mixing, check_target_selection
from network_checks import check_forecast_network

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.graph.graph_wavenet import (
    GraphWaveNetForecastNetwork,
    GraphWaveNetLayer,
    GraphWaveNetParameters,
)

MODELS: list[GraphWaveNetParameters] = [
    GraphWaveNetParameters(
        residual_channels=4, dilation_channels=4, skip_channels=8, end_channels=8, num_blocks=2, embedding_size=3
    ),
    GraphWaveNetParameters(
        residual_channels=4,
        dilation_channels=6,
        skip_channels=6,
        end_channels=6,
        kernel_size=3,
        num_blocks=1,
        num_layers=3,
        diffusion_order=1,
        dropout=0.0,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: GraphWaveNetParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_forecasts_only_the_targets() -> None:
    check_target_selection(MODELS[0])


def test_channels_mix_through_the_graph() -> None:
    check_channel_mixing(MODELS[0])


def test_receptive_field_and_last_layer() -> None:
    shape: ForecastShape = ForecastShape(input_length=5, input_channels=2, horizon=1, target_channels=(0,))
    network: GraphWaveNetForecastNetwork = GraphWaveNetForecastNetwork(GraphWaveNetParameters(), shape)
    assert network.padded_length == 13
    layers: list[GraphWaveNetLayer] = [layer for layer in network.layers if isinstance(layer, GraphWaveNetLayer)]
    assert [layer.graph_convolution is None for layer in layers] == [False] * 7 + [True]


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        GraphWaveNetParameters(kernel_size=1)
    with pytest.raises(ValueError, match="num_blocks"):
        GraphWaveNetParameters(num_blocks=0)
    with pytest.raises(ValueError, match="diffusion_order"):
        GraphWaveNetParameters(diffusion_order=0)
    with pytest.raises(ValueError, match="dropout"):
        GraphWaveNetParameters(dropout=-0.1)
