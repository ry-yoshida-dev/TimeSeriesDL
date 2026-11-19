import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models import EncoderHead, ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.convolutional.resnet import ResNetParameters, ResNetResidualBlock

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

MODELS: list[ResNetParameters] = [
    ResNetParameters(channel_sizes=(8, 16, 16)),
    ResNetParameters(channel_sizes=(6,), kernel_sizes=(4, 2), head=EncoderHead(pooling=TemporalPooling.LAST)),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ResNetParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: ResNetParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: ResNetParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_encoder_keeps_length() -> None:
    encoder = ResNetParameters(channel_sizes=(4, 4)).build_encoder(13, 2)
    features: Tensor = encoder(torch.randn(2, 13, 2))
    assert features.shape == (2, 13, 4)


def test_shortcut_projects_only_when_channels_change() -> None:
    expanding: ResNetResidualBlock = ResNetResidualBlock(3, 8, (8, 5, 3))
    keeping: ResNetResidualBlock = ResNetResidualBlock(8, 8, (8, 5, 3))
    assert not isinstance(expanding.shortcut, nn.BatchNorm1d)
    assert isinstance(keeping.shortcut, nn.BatchNorm1d)
    outputs: Tensor = keeping(torch.randn(2, 8, 11))
    assert outputs.shape == (2, 8, 11)
    assert bool((outputs >= 0).all())


def test_default_matches_paper() -> None:
    parameters: ResNetParameters = ResNetParameters()
    assert parameters.channel_sizes == (64, 128, 128)
    assert parameters.kernel_sizes == (8, 5, 3)
    assert parameters.head.pooling is TemporalPooling.MEAN


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="channel_sizes"):
        ResNetParameters(channel_sizes=())
    with pytest.raises(ValueError, match="channel_sizes"):
        ResNetParameters(channel_sizes=(-1,))
    with pytest.raises(ValueError, match="kernel_sizes"):
        ResNetParameters(kernel_sizes=())
    with pytest.raises(ValueError, match="kernel_sizes"):
        ResNetParameters(kernel_sizes=(3, 0))
