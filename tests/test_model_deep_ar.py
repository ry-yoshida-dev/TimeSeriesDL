import pytest
import torch
from covariate_network_checks import COVARIATE_SHAPE_CASES, check_covariate_forecast_network, make_covariates
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape, RecurrentCell
from time_series_dl.models.architectures.recurrent.deep_ar import DeepARParameters
from time_series_dl.models.interfaces import CovariateForecastShape, CovariateTensors

MODELS: list[DeepARParameters] = [
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
def test_forecast_network(model: DeepARParameters, shape: ForecastShape) -> None:
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


COVARIATE_SHAPE: CovariateForecastShape = CovariateForecastShape(
    forecast=SHAPE, known_channels=2, static_channels=1, static_cardinalities=(4,)
)


@pytest.mark.parametrize("shape", COVARIATE_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_covariate_forecast_network(model: DeepARParameters, shape: CovariateForecastShape) -> None:
    check_covariate_forecast_network(model.build_covariate_forecast_network, shape)


@pytest.mark.parametrize("cell", list(RecurrentCell))
def test_covariate_inference_feeds_back_the_location(cell: RecurrentCell) -> None:
    _ = torch.manual_seed(0)
    network = DeepARParameters(cell=cell, hidden_size=8).build_covariate_forecast_network(COVARIATE_SHAPE).eval()
    past: Tensor = torch.randn(2, 9, 3)
    covariates: CovariateTensors = make_covariates(COVARIATE_SHAPE, batch_size=2)
    with torch.no_grad():
        inferred: Tensor = network(past, covariates)
        forced: Tensor = network(past, covariates, inferred[..., SHAPE.location_index])
    assert torch.allclose(inferred, forced, atol=1e-5)


def test_known_covariates_of_a_step_reach_only_that_step_and_later() -> None:
    _ = torch.manual_seed(0)
    network = DeepARParameters(hidden_size=8).build_covariate_forecast_network(COVARIATE_SHAPE).eval()
    past: Tensor = torch.randn(2, 9, 3)
    covariates: CovariateTensors = make_covariates(COVARIATE_SHAPE, batch_size=2)
    known: Tensor = covariates.known.clone()
    known[:, 9 + 2] += 1.0
    changed = CovariateTensors(known=known, static=covariates.static, static_categories=covariates.static_categories)
    with torch.no_grad():
        outputs: Tensor = network(past, covariates)
        changed_outputs: Tensor = network(past, changed)
    assert torch.allclose(outputs[:, :2], changed_outputs[:, :2])
    assert not torch.allclose(outputs[:, 2], changed_outputs[:, 2])
