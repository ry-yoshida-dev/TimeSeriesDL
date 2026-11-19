import pytest
import torch
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.mlp.tide import TiDEParameters

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

MODELS: list[TiDEParameters] = [
    TiDEParameters(hidden_size=16, decoder_output_size=4, temporal_decoder_hidden_size=8),
    TiDEParameters(
        hidden_size=8,
        num_encoder_layers=1,
        num_decoder_layers=3,
        decoder_output_size=2,
        temporal_decoder_hidden_size=4,
        dropout=0.0,
        is_layer_normalized=False,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TiDEParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_each_forecast_reads_only_its_channel() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=3, horizon=4, target_channels=(2, 0))
    network = TiDEParameters(hidden_size=8, decoder_output_size=2, temporal_decoder_hidden_size=4)
    model = network.build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 10, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += 1.0
    changed[:, :, 1] -= 3.0
    with torch.no_grad():
        original: Tensor = model(past)
        outputs: Tensor = model(changed)
    assert torch.allclose(original[:, :, 1], outputs[:, :, 1])
    assert not torch.allclose(original[:, :, 0], outputs[:, :, 0])


def test_point_forecast_varies_with_layer_normalization() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=1, horizon=3, target_channels=(0,))
    model = TiDEParameters(hidden_size=8, decoder_output_size=2, temporal_decoder_hidden_size=4)
    network = model.build_forecast_network(shape).eval()
    with torch.no_grad():
        outputs: Tensor = network(torch.randn(5, 10, 1))
    assert float(outputs.std(dim=0).min()) > 0.0


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        TiDEParameters(hidden_size=0)
    with pytest.raises(ValueError, match="num_decoder_layers"):
        TiDEParameters(num_decoder_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        TiDEParameters(dropout=-0.1)
