import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, Mamba2Parameters, SequenceShape, SummaryShape

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

MODELS: list[Mamba2Parameters] = [
    Mamba2Parameters(model_size=8, state_size=4, head_size=4, chunk_length=4, num_layers=2),
    Mamba2Parameters(
        model_size=4, state_size=2, expansion=1, head_size=2, chunk_length=64, num_layers=1, is_bidirectional=True
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: Mamba2Parameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: Mamba2Parameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: Mamba2Parameters) -> None:
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
    with pytest.raises(ValueError, match="head_size"):
        Mamba2Parameters(model_size=8, expansion=1, head_size=3)
    with pytest.raises(ValueError, match="chunk_length"):
        Mamba2Parameters(chunk_length=0)
