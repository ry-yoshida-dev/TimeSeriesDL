import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastingModel, ForecastShape
from time_series_dl.models.architectures.mlp.n_hits import NHITSInterpolation, NHITSParameters, NHITSPooling, NHITSStack

SMALL_STACKS: tuple[NHITSStack, ...] = (
    NHITSStack(pooling_kernel_size=8, downsampling_factor=4, hidden_size=16),
    NHITSStack(pooling_kernel_size=2, downsampling_factor=2, num_blocks=2, hidden_size=16),
    NHITSStack(pooling_kernel_size=1, downsampling_factor=1, num_layers=1, hidden_size=16),
)

MODELS: list[ForecastingModel] = [
    NHITSParameters(stacks=SMALL_STACKS),
    NHITSParameters(stacks=SMALL_STACKS, pooling=NHITSPooling.AVERAGE, interpolation=NHITSInterpolation.NEAREST),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_default_stacks() -> None:
    parameters: NHITSParameters = NHITSParameters()
    assert [stack.pooling_kernel_size for stack in parameters.stacks] == [2, 2, 1]
    assert [stack.downsampling_factor for stack in parameters.stacks] == [4, 2, 1]


@pytest.mark.parametrize("interpolation", list(NHITSInterpolation))
def test_interpolation_basis_keeps_constants(interpolation: NHITSInterpolation) -> None:
    basis: Tensor = interpolation.build_basis(3, 10)
    assert basis.shape == (3, 10)
    assert torch.allclose(basis.sum(dim=0), torch.ones(10))


@pytest.mark.parametrize("pooling", list(NHITSPooling))
def test_pooling_keeps_partial_window(pooling: NHITSPooling) -> None:
    pooled: Tensor = pooling.build(4)(torch.arange(10, dtype=torch.float32).unsqueeze(0))
    assert pooled.shape == (1, 3)


def test_coarsest_stack_forecasts_a_constant() -> None:
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=1, horizon=6, target_channels=(0,))
    stack: NHITSStack = NHITSStack(pooling_kernel_size=4, downsampling_factor=6, hidden_size=8)
    network = NHITSParameters(stacks=(stack,)).build_forecast_network(shape).eval()
    with torch.no_grad():
        forecast: Tensor = network(torch.randn(3, 12, 1))[:, :, 0, 0]
    assert torch.allclose(forecast, forecast[:, :1].expand(-1, 6))


def test_channels_are_forecast_independently() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=3, horizon=4, target_channels=(1,))
    network = NHITSParameters(stacks=SMALL_STACKS).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 10, 3)
    changed: Tensor = past.clone()
    changed[:, :, [0, 2]] += 3.0
    with torch.no_grad():
        assert torch.allclose(network(past), network(changed))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="stacks"):
        NHITSParameters(stacks=())
    with pytest.raises(ValueError, match="pooling_kernel_size"):
        NHITSStack(pooling_kernel_size=0, downsampling_factor=1)
    with pytest.raises(ValueError, match="downsampling_factor"):
        NHITSStack(pooling_kernel_size=1, downsampling_factor=0)
    with pytest.raises(ValueError, match="kernel_size"):
        NHITSPooling.MAX.build(0)
    with pytest.raises(ValueError, match="num_coefficients"):
        NHITSInterpolation.LINEAR.build_basis(0, 4)
