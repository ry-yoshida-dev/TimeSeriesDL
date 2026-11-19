import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, S4Parameters, SequenceShape, SummaryShape

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

MODELS: list[S4Parameters] = [
    S4Parameters(model_size=8, state_size=8, num_layers=2),
    S4Parameters(model_size=4, state_size=5, num_layers=1, dropout=0.0, is_bidirectional=True),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: S4Parameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: S4Parameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: S4Parameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_encoder_is_causal() -> None:
    _ = torch.manual_seed(0)
    encoder = MODELS[0].build_encoder(12, 3).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, 8:] += 1.0
    with torch.no_grad():
        original: Tensor = encoder(past)
        outputs: Tensor = encoder(changed)
    assert outputs.shape == (2, 12, encoder.output_size)
    assert torch.allclose(original[:, :8], outputs[:, :8], atol=1e-5)
    assert not torch.allclose(original[:, 8:], outputs[:, 8:], atol=1e-5)


def test_bidirectional_encoder_reads_the_future() -> None:
    _ = torch.manual_seed(0)
    encoder = MODELS[1].build_encoder(12, 3).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, 8:] += 1.0
    with torch.no_grad():
        assert not torch.allclose(encoder(past)[:, :8], encoder(changed)[:, :8], atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        S4Parameters(model_size=0)
    with pytest.raises(ValueError, match="state_size"):
        S4Parameters(state_size=0)
    with pytest.raises(ValueError, match="min_step"):
        S4Parameters(min_step=0.2, max_step=0.1)
    with pytest.raises(ValueError, match="dropout"):
        S4Parameters(dropout=1.0)
