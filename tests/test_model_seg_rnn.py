import pytest
import torch
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.recurrent.seg_rnn import SegRNNParameters

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

MODELS: list[SegRNNParameters] = [
    SegRNNParameters(segment_length=4, hidden_size=8),
    SegRNNParameters(segment_length=5, hidden_size=7, dropout=0.0, is_channel_embedded=False),
    SegRNNParameters(segment_length=1, hidden_size=4),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: SegRNNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_each_forecast_reads_only_its_channel() -> None:
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=5, target_channels=(2, 0))
    network = SegRNNParameters(segment_length=4, hidden_size=8).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += 1.0
    changed[:, :, 1] -= 3.0
    with torch.no_grad():
        original: Tensor = network(past)
        outputs: Tensor = network(changed)
    assert torch.allclose(original[:, :, 1], outputs[:, :, 1])
    assert not torch.allclose(original[:, :, 0], outputs[:, :, 0])


def test_channel_embedding_distinguishes_identical_series() -> None:
    shape: ForecastShape = ForecastShape(input_length=8, input_channels=2, horizon=4, target_channels=(0, 1))
    past: Tensor = torch.randn(1, 8, 1).expand(-1, -1, 2)
    embedded = SegRNNParameters(segment_length=4, hidden_size=8).build_forecast_network(shape).eval()
    shared = SegRNNParameters(segment_length=4, hidden_size=8, is_channel_embedded=False)
    plain = shared.build_forecast_network(shape).eval()
    with torch.no_grad():
        assert not torch.allclose(embedded(past)[:, :, 0], embedded(past)[:, :, 1])
        assert torch.allclose(plain(past)[:, :, 0], plain(past)[:, :, 1])


def test_segment_longer_than_window_is_rejected() -> None:
    shape: ForecastShape = ForecastShape(input_length=24, input_channels=1, horizon=4, target_channels=(0,))
    with pytest.raises(ValueError, match="segment_length"):
        SegRNNParameters().build_forecast_network(shape)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="segment_length"):
        SegRNNParameters(segment_length=0)
    with pytest.raises(ValueError, match="hidden_size"):
        SegRNNParameters(hidden_size=0)
    with pytest.raises(ValueError, match="even"):
        SegRNNParameters(hidden_size=7)
    with pytest.raises(ValueError, match="dropout"):
        SegRNNParameters(dropout=1.0)
