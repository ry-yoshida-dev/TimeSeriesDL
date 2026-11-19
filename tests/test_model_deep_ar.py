import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastingModel, ForecastShape, RecurrentCell
from time_series_dl.models.architectures.recurrent.deep_ar import DeepARParameters

MODELS: list[ForecastingModel] = [
    DeepARParameters(hidden_size=8),
    DeepARParameters(cell=RecurrentCell.GRU, hidden_size=8, num_layers=1),
]

SHAPE: ForecastShape = ForecastShape(
    input_length=9,
    input_channels=3,
    horizon=5,
    target_channels=(2, 0),
    num_parameters=3,
    location_index=1,
    level_parameters=(0, 1, 2),
)


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_teacher_forcing_reads_only_previous_targets() -> None:
    _ = torch.manual_seed(0)
    network = DeepARParameters(hidden_size=8).build_forecast_network(SHAPE).eval()
    past: Tensor = torch.randn(2, 9, 3)
    future: Tensor = torch.randn(2, 5, 2)
    changed: Tensor = future.clone()
    changed[:, 2] += 1.0
    with torch.no_grad():
        outputs: Tensor = network(past, future)
        changed_outputs: Tensor = network(past, changed)
    assert torch.allclose(outputs[:, :3], changed_outputs[:, :3])
    assert not torch.allclose(outputs[:, 3:], changed_outputs[:, 3:])


@pytest.mark.parametrize("cell", list(RecurrentCell))
def test_inference_feeds_back_the_location(cell: RecurrentCell) -> None:
    _ = torch.manual_seed(0)
    network = DeepARParameters(cell=cell, hidden_size=8).build_forecast_network(SHAPE).eval()
    past: Tensor = torch.randn(2, 9, 3)
    with torch.no_grad():
        inferred: Tensor = network(past)
        forced: Tensor = network(past, inferred[..., SHAPE.location_index])
    assert torch.allclose(inferred, forced, atol=1e-5)


def test_rejects_future_targets_of_wrong_shape() -> None:
    network = DeepARParameters(hidden_size=8).build_forecast_network(SHAPE)
    with pytest.raises(ValueError, match="future_targets"):
        network(torch.randn(2, 9, 3), torch.randn(2, 4, 2))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        DeepARParameters(hidden_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        DeepARParameters(num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        DeepARParameters(dropout=1.0)
