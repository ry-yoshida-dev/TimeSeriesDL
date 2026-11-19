import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.convolutional.inception_time import (
    InceptionTimeModule,
    InceptionTimeParameters,
)

FORECAST_SHAPES: list[ForecastShape] = [
    ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
    ForecastShape(
        input_length=17,
        input_channels=4,
        horizon=5,
        target_channels=(2, 0),
        num_parameters=3,
        location_index=1,
        level_parameters=(0, 1, 2),
    ),
    ForecastShape(input_length=15, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

MODELS: list[InceptionTimeParameters] = [
    InceptionTimeParameters(num_filters=4, bottleneck_size=4, max_kernel_size=8),
    InceptionTimeParameters(
        num_filters=3, bottleneck_size=2, max_kernel_size=9, num_kernel_sizes=2, depth=4, residual_interval=2
    ),
    InceptionTimeParameters(num_filters=3, bottleneck_size=2, max_kernel_size=4, depth=2, is_residual=False),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: InceptionTimeParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: InceptionTimeParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: InceptionTimeParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_default_matches_paper() -> None:
    parameters: InceptionTimeParameters = InceptionTimeParameters()
    assert parameters.kernel_sizes == (40, 20, 10)
    assert parameters.module_size == 128
    assert parameters.head.pooling is TemporalPooling.MEAN
    encoder = parameters.build_encoder(30, 2)
    assert len(encoder.inception_modules) == 6
    assert len(encoder.shortcuts) == 2


@pytest.mark.parametrize("length", [1, 10, 21])
def test_encoder_keeps_length_with_even_kernels(length: int) -> None:
    encoder = InceptionTimeParameters(num_filters=2, bottleneck_size=2, max_kernel_size=12, depth=3).build_encoder(
        length, 3
    )
    features: Tensor = encoder(torch.randn(2, length, 3))
    assert features.shape == (2, length, 8)


def test_bottleneck_only_for_several_channels() -> None:
    assert isinstance(InceptionTimeModule(1, 4, 4, (8, 4)).bottleneck, nn.Identity)
    assert not isinstance(InceptionTimeModule(3, 4, 4, (8, 4)).bottleneck, nn.Identity)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_filters"):
        InceptionTimeParameters(num_filters=0)
    with pytest.raises(ValueError, match="depth"):
        InceptionTimeParameters(depth=0)
    with pytest.raises(ValueError, match="residual_interval"):
        InceptionTimeParameters(residual_interval=0)
    with pytest.raises(ValueError, match="max_kernel_size"):
        InceptionTimeParameters(max_kernel_size=3, num_kernel_sizes=3)
