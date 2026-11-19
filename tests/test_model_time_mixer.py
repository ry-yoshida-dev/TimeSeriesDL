import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.mlp.time_mixer import (
    TimeMixerParameters,
    TimeMixerSeasonMixing,
    TimeMixerTrendMixing,
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
    ForecastShape(input_length=16, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

MODELS: list[TimeMixerParameters] = [
    TimeMixerParameters(model_size=8, feedforward_size=8, kernel_size=5),
    TimeMixerParameters(
        model_size=4, feedforward_size=6, num_layers=1, num_down_sampling_layers=1, down_sampling_window=3, dropout=0.0
    ),
    TimeMixerParameters(model_size=4, feedforward_size=4, num_down_sampling_layers=0, is_affine=False),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TimeMixerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TimeMixerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TimeMixerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_forecast_reads_only_own_channel() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=16, input_channels=3, horizon=4, target_channels=(2, 0))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 16, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += torch.randn(2, 16)
    changed[:, :, 1] -= 3.0
    with torch.no_grad():
        original: Tensor = network(past)
        outputs: Tensor = network(changed)
    assert torch.allclose(original[:, :, 1], outputs[:, :, 1])
    assert not torch.allclose(original[:, :, 0], outputs[:, :, 0])


def test_forecast_follows_level_of_window() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(
        input_length=16, input_channels=1, horizon=3, target_channels=(0,), num_parameters=2, level_parameters=(0,)
    )
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 16, 1)
    with torch.no_grad():
        original: Tensor = network(past)
        shifted: Tensor = network(past + 5.0)
    assert torch.allclose(shifted[..., 0], original[..., 0] + 5.0, atol=1e-4)
    assert torch.allclose(shifted[..., 1], original[..., 1], atol=1e-4)


def test_scale_mixing_directions() -> None:
    lengths: tuple[int, ...] = (8, 4, 2)
    scales: list[Tensor] = [torch.randn(2, 3, length) for length in lengths]
    changed: list[Tensor] = [scales[0], scales[1], scales[2] + 1.0]
    with torch.no_grad():
        seasons: TimeMixerSeasonMixing = TimeMixerSeasonMixing(lengths)
        trends: TimeMixerTrendMixing = TimeMixerTrendMixing(lengths)
        original: list[Tensor] = seasons(scales)
        mixed: list[Tensor] = seasons(changed)
        assert torch.equal(original[0], mixed[0])
        assert torch.equal(original[1], mixed[1])
        assert not torch.allclose(trends(scales)[0], trends(changed)[0])
        assert [tensor.shape[-1] for tensor in trends(scales)] == list(lengths)


def test_scale_lengths_and_short_window() -> None:
    assert TimeMixerParameters().compute_scale_lengths(96) == (96, 48, 24, 12)
    shape: ForecastShape = ForecastShape(input_length=7, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="too short"):
        TimeMixerParameters().build_forecast_network(shape)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        TimeMixerParameters(model_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        TimeMixerParameters(num_layers=0)
    with pytest.raises(ValueError, match="num_down_sampling_layers"):
        TimeMixerParameters(num_down_sampling_layers=-1)
    with pytest.raises(ValueError, match="down_sampling_window"):
        TimeMixerParameters(down_sampling_window=1)
    with pytest.raises(ValueError, match="kernel_size"):
        TimeMixerParameters(kernel_size=4)
    with pytest.raises(ValueError, match="dropout"):
        TimeMixerParameters(dropout=1.0)
