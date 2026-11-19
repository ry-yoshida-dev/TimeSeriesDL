import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from graph_network_checks import check_channel_mixing, check_target_selection
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.graph.mtgnn import MTGNNDilatedInception, MTGNNEncoder, MTGNNParameters

MODELS: list[MTGNNParameters] = [
    MTGNNParameters(
        num_layers=2,
        convolution_channels=8,
        residual_channels=8,
        skip_channels=8,
        end_channels=8,
        embedding_size=4,
        num_neighbors=2,
    ),
    MTGNNParameters(
        num_layers=3,
        convolution_channels=4,
        residual_channels=4,
        skip_channels=6,
        end_channels=6,
        embedding_size=3,
        dilation_base=2,
        propagation_depth=1,
        dropout=0.0,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: MTGNNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: MTGNNParameters, channels: int) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=channels))


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MTGNNParameters, channels: int) -> None:
    check_summary_network(model, SummaryShape(input_length=40, input_channels=channels, output_size=4))


def test_forecasts_only_the_targets() -> None:
    check_target_selection(MODELS[0])


def test_channels_mix_through_the_graph() -> None:
    check_channel_mixing(MODELS[0])


@pytest.mark.parametrize(("dilation_base", "padded_length"), [(1, 19), (2, 43)])
def test_short_windows_are_padded_to_the_receptive_field(dilation_base: int, padded_length: int) -> None:
    model: MTGNNParameters = MTGNNParameters(convolution_channels=4, dilation_base=dilation_base)
    assert MTGNNEncoder(model, 2, 5).padded_length == padded_length
    assert MTGNNEncoder(model, 2, 60).padded_length == 60


def test_dilated_inception_crops_to_the_longest_kernel() -> None:
    inception: MTGNNDilatedInception = MTGNNDilatedInception(2, 8, dilation=2)
    assert inception(torch.randn(3, 2, 5, 20)).shape == (3, 8, 5, 20 - 6 * 2)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="convolution_channels"):
        MTGNNParameters(convolution_channels=6)
    with pytest.raises(ValueError, match="num_neighbors"):
        MTGNNParameters(num_neighbors=0)
    with pytest.raises(ValueError, match="retain_ratio"):
        MTGNNParameters(retain_ratio=1.5)
    with pytest.raises(ValueError, match="propagation_depth"):
        MTGNNParameters(propagation_depth=0)
    with pytest.raises(ValueError, match="dropout"):
        MTGNNParameters(dropout=1.0)
