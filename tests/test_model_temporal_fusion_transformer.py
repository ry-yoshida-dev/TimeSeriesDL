import pytest
import torch
from covariate_network_checks import COVARIATE_SHAPE_CASES, check_covariate_forecast_network, make_covariates
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.transformer.temporal_fusion_transformer import (
    GatedResidualNetwork,
    TemporalFusionInterpretation,
    TemporalFusionTransformerCovariateForecastNetwork,
    TemporalFusionTransformerForecastNetwork,
    TemporalFusionTransformerParameters,
)
from time_series_dl.models.interfaces import CovariateForecastShape, CovariateTensors

MODELS: list[TemporalFusionTransformerParameters] = [
    TemporalFusionTransformerParameters(hidden_size=8, num_heads=2),
    TemporalFusionTransformerParameters(hidden_size=12, num_heads=3, num_lstm_layers=2, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TemporalFusionTransformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_interpretation_weights() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=7, input_channels=3, horizon=4, target_channels=(1,))
    network = TemporalFusionTransformerForecastNetwork(
        TemporalFusionTransformerParameters(hidden_size=8, num_heads=2), shape
    ).eval()
    with torch.no_grad():
        interpretation: TemporalFusionInterpretation = network.interpret(torch.randn(2, 7, 3))
    assert interpretation.variable_weights.shape == (2, 7, 3)
    assert torch.allclose(interpretation.variable_weights.sum(dim=-1), torch.ones(2, 7))
    attention: Tensor = interpretation.attention_weights
    assert attention.shape == (2, 4, 11)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 4))
    is_future: Tensor = torch.arange(11).unsqueeze(0) > torch.arange(4).unsqueeze(1) + 7
    assert bool((attention[:, is_future] == 0.0).all())
    assert bool((attention[:, ~is_future] > 0.0).all())


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        TemporalFusionTransformerParameters(hidden_size=0)
    with pytest.raises(ValueError, match="divisible"):
        TemporalFusionTransformerParameters(hidden_size=10, num_heads=4)
    with pytest.raises(ValueError, match="num_lstm_layers"):
        TemporalFusionTransformerParameters(num_lstm_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        TemporalFusionTransformerParameters(dropout=-0.1)


@pytest.mark.parametrize("shape", COVARIATE_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_covariate_forecast_network(model: TemporalFusionTransformerParameters, shape: CovariateForecastShape) -> None:
    check_covariate_forecast_network(model.build_covariate_forecast_network, shape)


def test_covariate_interpretation_weights() -> None:
    _ = torch.manual_seed(0)
    shape = CovariateForecastShape(
        forecast=ForecastShape(input_length=7, input_channels=3, horizon=4, target_channels=(1,)),
        known_channels=2,
        static_channels=1,
        static_cardinalities=(5, 2),
    )
    network = TemporalFusionTransformerCovariateForecastNetwork(
        TemporalFusionTransformerParameters(hidden_size=8, num_heads=2, num_lstm_layers=2), shape
    ).eval()
    with torch.no_grad():
        interpretation: TemporalFusionInterpretation = network.interpret(
            torch.randn(2, 7, 3), make_covariates(shape, batch_size=2)
        )
    assert interpretation.variable_weights.shape == (2, 7, 5)
    assert interpretation.attention_weights.shape == (2, 4, 11)
    future_weights = interpretation.future_variable_weights
    static_weights = interpretation.static_variable_weights
    assert future_weights is not None and future_weights.shape == (2, 4, 2)
    assert static_weights is not None and static_weights.shape == (2, 3)
    assert torch.allclose(future_weights.sum(dim=-1), torch.ones(2, 4))
    assert torch.allclose(static_weights.sum(dim=-1), torch.ones(2))


def test_known_covariates_of_a_step_do_not_reach_earlier_steps() -> None:
    _ = torch.manual_seed(0)
    shape = CovariateForecastShape(
        forecast=ForecastShape(input_length=6, input_channels=1, horizon=4, target_channels=(0,)), known_channels=1
    )
    network = TemporalFusionTransformerParameters(hidden_size=8, num_heads=2).build_covariate_forecast_network(shape)
    _ = network.eval()
    past: Tensor = torch.randn(2, 6, 1)
    covariates: CovariateTensors = make_covariates(shape, batch_size=2)
    known: Tensor = covariates.known.clone()
    known[:, 6 + 2] += 1.0
    changed = CovariateTensors(known=known, static=covariates.static, static_categories=covariates.static_categories)
    with torch.no_grad():
        outputs: Tensor = network(past, covariates)
        changed_outputs: Tensor = network(past, changed)
    assert torch.allclose(outputs[:, :2], changed_outputs[:, :2])
    assert not torch.allclose(outputs[:, 2:], changed_outputs[:, 2:])


def test_gated_residual_network_requires_its_context() -> None:
    network = GatedResidualNetwork(4, 4, 4, 0.0, context_size=3)
    with pytest.raises(ValueError, match="context"):
        network(torch.randn(2, 5, 4))
    assert network(torch.randn(2, 5, 4), torch.randn(2, 3)).shape == (2, 5, 4)
