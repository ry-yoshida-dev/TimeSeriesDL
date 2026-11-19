import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastingModel, ForecastShape
from time_series_dl.models.architectures.mlp.n_beats import NBEATSBlockKind, NBEATSParameters, NBEATSStack

MODELS: list[ForecastingModel] = [
    NBEATSParameters.generic(num_stacks=3, num_layers=2, hidden_size=16),
    NBEATSParameters.interpretable(num_blocks=2, num_layers=2, trend_hidden_size=8, seasonality_hidden_size=16),
    NBEATSParameters(
        stacks=(
            NBEATSStack(kind=NBEATSBlockKind.SEASONALITY, num_blocks=2, num_layers=1, hidden_size=8),
            NBEATSStack(kind=NBEATSBlockKind.TREND, num_layers=1, hidden_size=8),
        ),
        polynomial_degree=1,
        num_harmonics=2,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_default_is_generic_architecture() -> None:
    parameters: NBEATSParameters = NBEATSParameters()
    assert parameters == NBEATSParameters.generic()
    assert len(parameters.stacks) == 30
    assert all(stack.kind is NBEATSBlockKind.GENERIC for stack in parameters.stacks)


@pytest.mark.parametrize("kind", list(NBEATSBlockKind))
def test_bases_have_matching_lengths(kind: NBEATSBlockKind) -> None:
    backcast: Tensor
    forecast: Tensor
    backcast, forecast = kind.build_bases(11, 6, 2, 1)
    assert backcast.shape[1] == 11
    assert forecast.shape[1] == 6
    is_interpretable: bool = kind is not NBEATSBlockKind.GENERIC
    assert (backcast.shape[0] == forecast.shape[0]) == is_interpretable
    assert bool(torch.isfinite(backcast).all() and torch.isfinite(forecast).all())


def test_seasonality_basis_is_periodic_over_the_horizon() -> None:
    backcast: Tensor
    forecast: Tensor
    backcast, forecast = NBEATSBlockKind.SEASONALITY.build_bases(12, 6, 3, 1)
    assert torch.allclose(backcast[:, 6:], forecast, atol=1e-5)
    assert forecast.shape[0] == 5


def test_trend_forecast_is_a_polynomial() -> None:
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=1, horizon=8, target_channels=(0,))
    parameters: NBEATSParameters = NBEATSParameters(
        stacks=(NBEATSStack(kind=NBEATSBlockKind.TREND, num_blocks=2, num_layers=2, hidden_size=8),),
        polynomial_degree=2,
    )
    network = parameters.build_forecast_network(shape).eval()
    with torch.no_grad():
        forecast: Tensor = network(torch.randn(4, 12, 1))[:, :, 0, 0]
    assert torch.allclose(forecast.diff(n=3, dim=1), torch.zeros(4, 5), atol=1e-4)


def test_shared_stack_reuses_one_block() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=1, horizon=4, target_channels=(0,))

    def count(is_weight_shared: bool) -> int:
        stack: NBEATSStack = NBEATSStack(num_blocks=3, num_layers=1, hidden_size=8, is_weight_shared=is_weight_shared)
        network = NBEATSParameters(stacks=(stack,)).build_forecast_network(shape)
        return sum(parameter.numel() for parameter in network.parameters())

    assert count(False) == 3 * count(True)


def test_channels_are_forecast_independently() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=3, horizon=4, target_channels=(2, 0))
    network = NBEATSParameters.generic(num_stacks=2, num_layers=1, hidden_size=8).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 10, 3)
    changed: Tensor = past.clone()
    changed[:, :, 1] += 5.0
    changed[:, :, 0] += 1.0
    with torch.no_grad():
        outputs: Tensor = network(past)
        changed_outputs: Tensor = network(changed)
    assert torch.allclose(outputs[:, :, 0], changed_outputs[:, :, 0])
    assert not torch.allclose(outputs[:, :, 1], changed_outputs[:, :, 1])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="stacks"):
        NBEATSParameters(stacks=())
    with pytest.raises(ValueError, match="polynomial_degree"):
        NBEATSParameters(polynomial_degree=-1)
    with pytest.raises(ValueError, match="num_harmonics"):
        NBEATSParameters(num_harmonics=0)
    with pytest.raises(ValueError, match="num_stacks"):
        NBEATSParameters.generic(num_stacks=0)
    with pytest.raises(ValueError, match="num_blocks"):
        NBEATSStack(num_blocks=0)
    with pytest.raises(ValueError, match="hidden_size"):
        NBEATSParameters.interpretable(trend_hidden_size=0)
