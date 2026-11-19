import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.convolutional.lite import (
    LITECustomFilters,
    LITEInceptionModule,
    LITEParameters,
    LITESeparableConvolutionBlock,
)
from time_series_dl.models.components.layers import SameLengthConvolution

MODELS: list[LITEParameters] = [
    LITEParameters(num_filters=4, kernel_size=9),
    LITEParameters(num_filters=3, kernel_size=8, is_custom_filtered=False, is_dilated=False, is_multiplexed=False),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: LITEParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: LITEParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: LITEParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_default_matches_the_paper() -> None:
    parameters: LITEParameters = LITEParameters()
    assert parameters.head.pooling is TemporalPooling.MEAN
    encoder = parameters.build_encoder(50, 2)
    branches: list[SameLengthConvolution] = [
        branch for branch in encoder.inception.branches if isinstance(branch, SameLengthConvolution)
    ]
    assert [branch.convolution.kernel_size[0] for branch in branches] == [41, 20, 10]
    assert encoder.inception.output_size == 3 * 32 + 17
    blocks: list[LITESeparableConvolutionBlock] = [
        block for block in encoder.blocks if isinstance(block, LITESeparableConvolutionBlock)
    ]
    assert [block.depthwise.kernel_size[0] for block in blocks] == [20, 10]
    assert [block.depthwise.dilation[0] for block in blocks] == [2, 4]
    assert encoder.output_size == 32
    univariate = parameters.build_encoder(50, 1)
    assert sum(parameter.numel() for parameter in univariate.parameters()) == 9846


def test_custom_filters_are_fixed_and_match_the_official_ones() -> None:
    filters: LITECustomFilters = LITECustomFilters(2)
    assert filters.num_filters == 17
    assert not list(filters.parameters())
    center: int = (filters.width - 1) // 2
    increasing: Tensor = filters.weights[0, 0, center : center + 2]
    torch.testing.assert_close(increasing, torch.tensor([-1.0, 1.0]))
    decreasing: Tensor = filters.weights[7, 0, center - 1 : center + 3]
    torch.testing.assert_close(decreasing, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    peak: Tensor = filters.weights[12, 0, center - 2 : center + 4]
    torch.testing.assert_close(peak, torch.tensor([-1.0, -1.0, 2.0, 2.0, -1.0, -1.0]))
    torch.testing.assert_close(filters.weights[:, 0], filters.weights[:, 1])


def test_custom_filters_match_keras_same_convolution() -> None:
    filters: LITECustomFilters = LITECustomFilters(1)
    values: Tensor = torch.randn(2, 1, 30)
    outputs: Tensor = filters(values)
    kernel: Tensor = torch.tensor([1.0, -1.0, 1.0, -1.0])
    padded: Tensor = torch.nn.functional.pad(values, (1, 2))
    expected: Tensor = torch.relu(torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, 4)))
    torch.testing.assert_close(outputs[:, 7:8], expected)


def test_inception_module_sizes() -> None:
    module: LITEInceptionModule = LITEInceptionModule(3, 4, 8, is_multiplexed=False, is_custom_filtered=False)
    assert len(module.branches) == 1
    assert module(torch.randn(2, 3, 11)).shape == (2, 12, 11)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_filters"):
        LITEParameters(num_filters=0)
    with pytest.raises(ValueError, match="kernel_size"):
        LITEParameters(kernel_size=3)
