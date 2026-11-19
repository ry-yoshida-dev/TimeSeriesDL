import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.mlp.time_mixer_plus_plus import (
    TimeMixerPlusPlusParameters,
    TimeMixerPlusPlusTimeImaging,
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
    ForecastShape(input_length=15, input_channels=3, horizon=7, target_channels=(2, 1), num_parameters=2),
    ForecastShape(input_length=16, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

MODELS: list[TimeMixerPlusPlusParameters] = [
    TimeMixerPlusPlusParameters(model_size=8, num_heads=2, num_layers=1, num_down_samplings=2, top_k=2),
    TimeMixerPlusPlusParameters(model_size=4, num_heads=1, num_layers=2, num_down_samplings=1, top_k=3, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TimeMixerPlusPlusParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TimeMixerPlusPlusParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TimeMixerPlusPlusParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_target_subset_selects_from_every_channel_forecast() -> None:
    parameters: TimeMixerPlusPlusParameters = MODELS[1]
    every: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(0, 1, 2))
    subset: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(2, 0))
    _ = torch.manual_seed(0)
    full_network = parameters.build_forecast_network(every).eval()
    _ = torch.manual_seed(0)
    subset_network = parameters.build_forecast_network(subset).eval()
    past: Tensor = torch.randn(2, 12, 3)
    with torch.no_grad():
        assert torch.allclose(subset_network(past), full_network(past)[:, :, [2, 0]])


def test_forecast_follows_level_of_window() -> None:
    shape: ForecastShape = ForecastShape(input_length=16, input_channels=2, horizon=3, target_channels=(0, 1))
    _ = torch.manual_seed(0)
    network = MODELS[1].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 16, 2)
    with torch.no_grad():
        assert torch.allclose(network(past + 5.0), network(past) + 5.0, atol=1e-4)


def test_time_imaging_folds_by_period_and_unfolds_back() -> None:
    imaging: TimeMixerPlusPlusTimeImaging = TimeMixerPlusPlusTimeImaging(top_k=2)
    tokens: Tensor = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
    image: Tensor = imaging.to_image(tokens, 3)
    assert image.shape == (1, 2, 3, 3)
    # Column j holds the j-th period, row i its i-th phase; the last period is zero-padded.
    assert torch.equal(image[0, 0, :, 1], tokens[0, 3:6, 0])
    assert torch.equal(image[0, 1, :, 2], torch.tensor([13.0, 0.0, 0.0]))
    assert torch.equal(imaging.to_sequence(image, 7), tokens)


def test_time_imaging_measures_periods_in_input_steps() -> None:
    imaging: TimeMixerPlusPlusTimeImaging = TimeMixerPlusPlusTimeImaging(top_k=1)
    steps: Tensor = torch.arange(8, dtype=torch.float32)
    coarsest: Tensor = torch.cos(2.0 * torch.pi * 2.0 * steps / 8.0)[None, :, None]
    periods: list[int]
    weights: Tensor
    periods, weights = imaging.detect(coarsest, input_length=32)
    assert periods == [16]
    assert torch.allclose(weights, torch.ones(1, 1))


def test_parameters_reject_invalid_values() -> None:
    for name in ("model_size", "num_heads", "num_layers", "num_down_samplings", "top_k"):
        with pytest.raises(ValueError, match=name):
            TimeMixerPlusPlusParameters(**{name: 0})
    with pytest.raises(ValueError, match="divide"):
        TimeMixerPlusPlusParameters(model_size=6, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        TimeMixerPlusPlusParameters(dropout=1.0)
    with pytest.raises(ValueError, match="too short"):
        TimeMixerPlusPlusParameters(num_down_samplings=3).build_forecast_network(
            ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
        )
