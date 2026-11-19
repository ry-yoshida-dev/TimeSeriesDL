import math

import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.convolutional.times_net import PeriodDetector, TimesNetParameters

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

MODELS: list[TimesNetParameters] = [
    TimesNetParameters(model_size=8, feedforward_size=8, num_layers=2, top_k=3, num_kernels=3),
    TimesNetParameters(model_size=4, feedforward_size=6, num_layers=1, top_k=1, num_kernels=2, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TimesNetParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("length", [19, 2, 1])
@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TimesNetParameters, length: int) -> None:
    check_reconstruction_network(model, SequenceShape(length=length, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TimesNetParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_detector_finds_dominant_periods() -> None:
    time: Tensor = torch.arange(64, dtype=torch.float32)
    wave: Tensor = torch.sin(2 * math.pi * time / 8) + 0.5 * torch.sin(2 * math.pi * time / 32)
    values: Tensor = wave.reshape(1, 64, 1).expand(2, -1, 3)
    periods: list[int]
    amplitudes: Tensor
    periods, amplitudes = PeriodDetector(top_k=2).detect(values)
    assert periods == [8, 32]
    assert amplitudes.shape == (2, 2)
    assert bool((amplitudes[:, 0] > amplitudes[:, 1]).all())


@pytest.mark.parametrize(("length", "top_k", "count"), [(1, 3, 1), (2, 3, 1), (5, 3, 2), (9, 2, 2)])
def test_detector_keeps_periods_within_length(length: int, top_k: int, count: int) -> None:
    periods: list[int]
    amplitudes: Tensor
    periods, amplitudes = PeriodDetector(top_k=top_k).detect(torch.zeros(2, length, 3))
    assert len(periods) == count
    assert amplitudes.shape == (2, count)
    assert all(1 <= period <= length for period in periods)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="top_k"):
        TimesNetParameters(top_k=0)
    with pytest.raises(ValueError, match="num_kernels"):
        TimesNetParameters(num_kernels=0)
    with pytest.raises(ValueError, match="dropout"):
        TimesNetParameters(dropout=1.0)
    with pytest.raises(ValueError, match="top_k"):
        PeriodDetector(top_k=-1)
