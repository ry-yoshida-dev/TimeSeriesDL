import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import EncoderHead, ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.convolutional.fcn import FCNParameters

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

MODELS: list[FCNParameters] = [
    FCNParameters(channel_sizes=(8, 16, 8)),
    FCNParameters(
        channel_sizes=(4, 6), kernel_sizes=(4, 2), head=EncoderHead(pooling=TemporalPooling.MAX, dropout=0.1)
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FCNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: FCNParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: FCNParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


@pytest.mark.parametrize("length", [1, 2, 13, 16])
def test_encoder_keeps_length_with_even_kernels(length: int) -> None:
    encoder = FCNParameters(channel_sizes=(4, 5, 6), kernel_sizes=(8, 5, 2)).build_encoder(length, 2)
    features: Tensor = encoder(torch.randn(2, length, 2))
    assert features.shape == (2, length, 6)
    assert (encoder.output_length, encoder.output_size) == (length, 6)


def test_default_matches_paper() -> None:
    parameters: FCNParameters = FCNParameters()
    assert parameters.channel_sizes == (128, 256, 128)
    assert parameters.kernel_sizes == (8, 5, 3)
    assert parameters.head.pooling is TemporalPooling.MEAN


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="channel_sizes"):
        FCNParameters(channel_sizes=())
    with pytest.raises(ValueError, match="channel_sizes"):
        FCNParameters(channel_sizes=(8, 0, 8))
    with pytest.raises(ValueError, match="kernel_sizes"):
        FCNParameters(kernel_sizes=(8, 0, 3))
    with pytest.raises(ValueError, match="one length per block"):
        FCNParameters(kernel_sizes=(8, 5))
