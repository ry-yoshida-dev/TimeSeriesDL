import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.mlp.ts_mixer import TSMixerNormalization, TSMixerParameters

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

MODELS: list[TSMixerParameters] = [
    TSMixerParameters(hidden_size=8),
    TSMixerParameters(
        num_blocks=1, hidden_size=4, dropout=0.0, normalization=TSMixerNormalization.LAYER, pooling=TemporalPooling.MEAN
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TSMixerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TSMixerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TSMixerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_target_subset_selects_from_every_channel_forecast() -> None:
    parameters: TSMixerParameters = TSMixerParameters(hidden_size=8, dropout=0.0)
    every: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(0, 1, 2))
    subset: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(2, 0))
    _ = torch.manual_seed(0)
    full_network = parameters.build_forecast_network(every).eval()
    _ = torch.manual_seed(0)
    subset_network = parameters.build_forecast_network(subset).eval()
    past: Tensor = torch.randn(2, 12, 3)
    with torch.no_grad():
        assert torch.allclose(subset_network(past), full_network(past)[:, :, [2, 0]])


def test_mixer_mixes_channels() -> None:
    shape: ForecastShape = ForecastShape(input_length=8, input_channels=2, horizon=3, target_channels=(0,))
    network = TSMixerParameters(hidden_size=8, dropout=0.0).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(4, 8, 2)
    changed: Tensor = past.clone()
    changed[:, :, 1] += 1.0
    with torch.no_grad():
        assert not torch.allclose(network(past), network(changed))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        TSMixerParameters(num_blocks=0)
    with pytest.raises(ValueError, match="hidden_size"):
        TSMixerParameters(hidden_size=0)
    with pytest.raises(ValueError, match="dropout"):
        TSMixerParameters(dropout=1.0)
