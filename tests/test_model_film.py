import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import FiLMParameters, ForecastShape
from time_series_dl.models.architectures.state_space.film import FiLMLegendreProjection

MODELS: list[FiLMParameters] = [
    FiLMParameters(state_size=16, num_modes=4),
    FiLMParameters(state_size=8, scales=(2,), num_modes=1),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FiLMParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_projection_reconstructs_newest_steps_of_window() -> None:
    projection: FiLMLegendreProjection = FiLMLegendreProjection(
        state_size=32, window_length=48, input_length=60, horizon=8
    )
    steps: Tensor = torch.arange(60, dtype=torch.float32)
    values: Tensor = (torch.sin(steps / 6.0) + steps / 30.0)[None, None, :]
    states: Tensor = projection.encode(values)
    assert states.shape == (1, 1, 32, 48)
    assert torch.allclose(projection.reconstruct(states[..., -1])[0, 0], values[0, 0, -8:], atol=0.05)


def test_each_forecast_reads_only_its_channel() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=16, input_channels=3, horizon=4, target_channels=(2, 0))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 16, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += torch.randn(2, 16)
    changed[:, :, 1] -= 3.0
    with torch.no_grad():
        original: Tensor = network(past)
        outputs: Tensor = network(changed)
    assert torch.allclose(original[:, :, 1], outputs[:, :, 1])
    assert not torch.allclose(original[:, :, 0], outputs[:, :, 0])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="state_size"):
        FiLMParameters(state_size=0)
    with pytest.raises(ValueError, match="num_modes"):
        FiLMParameters(num_modes=0)
    with pytest.raises(ValueError, match="scales"):
        FiLMParameters(scales=())
    with pytest.raises(ValueError, match="scales"):
        FiLMParameters(scales=(1, 1))
    with pytest.raises(ValueError, match="horizon"):
        FiLMLegendreProjection(state_size=4, window_length=4, input_length=8, horizon=5)
