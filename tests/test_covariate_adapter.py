from functools import partial

import pytest
import torch
from covariate_network_checks import COVARIATE_SHAPE_CASES, check_covariate_forecast_network, make_covariates
from torch import Tensor

from time_series_dl.models import DLinearParameters, ForecastingModel, ForecastShape, MLPParameters, RNNParameters
from time_series_dl.models.components.covariates import CovariateAdapterNetwork, CovariateAdapterParameters
from time_series_dl.models.interfaces import CovariateForecastShape, CovariateTensors

MODELS: list[ForecastingModel] = [
    MLPParameters(hidden_sizes=(8,)),
    DLinearParameters(kernel_size=3),
    RNNParameters(hidden_size=8, num_layers=1),
]

SHAPE: CovariateForecastShape = CovariateForecastShape(
    forecast=ForecastShape(input_length=8, input_channels=2, horizon=3, target_channels=(1,)),
    known_channels=2,
    static_channels=1,
    static_cardinalities=(3,),
)


@pytest.mark.parametrize("shape", COVARIATE_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_adapter_network(model: ForecastingModel, shape: CovariateForecastShape) -> None:
    adapter = CovariateAdapterParameters(hidden_size=8, dropout=0.1)
    check_covariate_forecast_network(partial(adapter.build_network, model), shape)


def test_base_network_reads_past_known_covariates_as_extra_channels() -> None:
    _ = torch.manual_seed(0)
    network = CovariateAdapterParameters(hidden_size=8).build_network(MLPParameters(hidden_sizes=(8,)), SHAPE)
    assert isinstance(network, CovariateAdapterNetwork)
    _ = network.eval()
    past: Tensor = torch.randn(2, 8, 2)
    covariates: CovariateTensors = make_covariates(SHAPE, batch_size=2)
    known: Tensor = covariates.known.clone()
    known[:, :8] += 1.0
    changed = CovariateTensors(known=known, static=covariates.static, static_categories=covariates.static_categories)
    with torch.no_grad():
        base_inputs: Tensor = torch.cat([past, covariates.known[:, :8]], dim=2)
        assert network.base(base_inputs).shape == (2, 3, 1, 1)
        assert not torch.allclose(network(past, covariates), network(past, changed))


def test_correction_reads_each_step_covariates_only() -> None:
    _ = torch.manual_seed(0)
    network = CovariateAdapterParameters(hidden_size=8).build_network(MLPParameters(hidden_sizes=(8,)), SHAPE).eval()
    past: Tensor = torch.randn(2, 8, 2)
    covariates: CovariateTensors = make_covariates(SHAPE, batch_size=2)
    known: Tensor = covariates.known.clone()
    known[:, 8 + 1] += 1.0
    changed = CovariateTensors(known=known, static=covariates.static, static_categories=covariates.static_categories)
    with torch.no_grad():
        outputs: Tensor = network(past, covariates)
        changed_outputs: Tensor = network(past, changed)
    assert torch.allclose(outputs[:, [0, 2]], changed_outputs[:, [0, 2]])
    assert not torch.allclose(outputs[:, 1], changed_outputs[:, 1])


def test_adapter_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        CovariateAdapterParameters(hidden_size=0)
    with pytest.raises(ValueError, match="dropout"):
        CovariateAdapterParameters(dropout=1.0)


def test_covariate_shape_and_tensors_validate() -> None:
    forecast = ForecastShape(input_length=4, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="needs known or static"):
        CovariateForecastShape(forecast=forecast)
    with pytest.raises(ValueError, match="static_cardinalities"):
        CovariateForecastShape(forecast=forecast, static_cardinalities=(0,))
    with pytest.raises(ValueError, match="known_channels"):
        CovariateForecastShape(forecast=forecast, known_channels=-1, static_channels=1)
    with pytest.raises(TypeError, match="int32 or int64"):
        CovariateTensors(known=torch.zeros(2, 6, 1), static=torch.zeros(2, 0), static_categories=torch.zeros(2, 1))
    with pytest.raises(ValueError, match="batch size"):
        CovariateTensors(
            known=torch.zeros(2, 6, 1), static=torch.zeros(3, 0), static_categories=torch.zeros(2, 0, dtype=torch.int64)
        )
