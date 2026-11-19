import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.convolutional.micn import (
    MICNIsometricLayer,
    MICNMultiScaleDecomposition,
    MICNParameters,
)

MODELS: list[MICNParameters] = [
    MICNParameters(model_size=8, kernel_sizes=(4, 6)),
    MICNParameters(model_size=6, num_layers=2, kernel_sizes=(3, 5, 12), dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: MICNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("length", [19, 2, 1])
@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: MICNParameters, length: int) -> None:
    check_reconstruction_network(model, SequenceShape(length=length, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MICNParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_default_matches_the_paper() -> None:
    parameters: MICNParameters = MICNParameters()
    assert parameters.kernel_sizes == (12, 16)
    assert parameters.decomposition_sizes == (13, 17)
    layer: MICNIsometricLayer = MICNIsometricLayer(MICNParameters(model_size=4), 96 + 96)
    sizes: list[int] = []
    for convolution in layer.isometric:
        assert isinstance(convolution, nn.Conv1d)
        sizes.append(convolution.kernel_size[0])
    assert sizes == [17, 13]


@pytest.mark.parametrize("length", [1, 7, 24, 50])
def test_isometric_kernel_spans_the_downsampled_sequence(length: int) -> None:
    layer: MICNIsometricLayer = MICNIsometricLayer(MICNParameters(model_size=4, kernel_sizes=(2, 3, 8)), length)
    for downsampling, isometric in zip(layer.downsampling, layer.isometric, strict=True):
        assert isinstance(downsampling, nn.Conv1d)
        assert isinstance(isometric, nn.Conv1d)
        downsampled: Tensor = downsampling(torch.zeros(1, 4, length))
        assert isometric.kernel_size[0] == downsampled.shape[-1]
    assert layer(torch.randn(2, length, 4)).shape == (2, length, 4)


def test_multiscale_decomposition_averages_the_trends() -> None:
    values: Tensor = torch.randn(2, 20, 3)
    decomposition: MICNMultiScaleDecomposition = MICNMultiScaleDecomposition((3, 5))
    seasonal: Tensor
    trend: Tensor
    seasonal, trend = decomposition(values)
    first: Tensor = decomposition.decompositions[0](values)[1]
    second: Tensor = decomposition.decompositions[1](values)[1]
    torch.testing.assert_close(trend, (first + second) / 2)
    torch.testing.assert_close(seasonal + trend, values)


def test_trend_is_added_to_level_parameters_only() -> None:
    shape: ForecastShape = ForecastShape(
        input_length=12, input_channels=2, horizon=3, target_channels=(1,), num_parameters=2
    )
    network = MICNParameters(model_size=4, kernel_sizes=(4,)).build_forecast_network(shape).eval()
    values: Tensor = torch.randn(1, 12, 2)
    shifted: Tensor = values.clone()
    shifted[:, :, 1] += 10.0
    with torch.no_grad():
        difference: Tensor = network(shifted) - network(values)
    torch.testing.assert_close(difference[..., 1], torch.zeros(1, 3, 1), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(difference[..., 0], torch.full((1, 3, 1), 10.0 * 12 / 3), atol=1e-3, rtol=0.0)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        MICNParameters(model_size=0)
    with pytest.raises(ValueError, match="kernel_sizes"):
        MICNParameters(kernel_sizes=())
    with pytest.raises(ValueError, match="kernel_sizes"):
        MICNParameters(kernel_sizes=(0,))
    with pytest.raises(ValueError, match="dropout"):
        MICNParameters(dropout=1.0)
