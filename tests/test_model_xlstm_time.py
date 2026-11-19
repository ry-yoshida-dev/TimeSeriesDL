import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.recurrent.xlstm import XLSTMStackParameters
from time_series_dl.models.architectures.recurrent.xlstm_time import XLSTMTimeForecastNetwork, XLSTMTimeParameters

MODELS: list[XLSTMTimeParameters] = [
    XLSTMTimeParameters(
        moving_average_size=5,
        model_size=8,
        stack=XLSTMStackParameters(num_blocks=1, slstm_block_indices=(0,), slstm_num_heads=2),
    ),
    XLSTMTimeParameters(
        moving_average_size=3,
        model_size=4,
        stack=XLSTMStackParameters(num_blocks=2, slstm_block_indices=(), projection_block_size=2),
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: XLSTMTimeParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_targets_select_channels_of_the_full_forecast() -> None:
    every_channel = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(0, 1, 2))
    some_channels = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(2, 0))
    full = MODELS[0].build_forecast_network(every_channel).eval()
    selected = MODELS[0].build_forecast_network(some_channels).eval()
    _ = selected.load_state_dict(full.state_dict())
    past: Tensor = torch.randn(2, 12, 3)
    with torch.no_grad():
        assert torch.allclose(selected(past), full(past)[:, :, [2, 0]], atol=1e-6)


def test_linear_forecasts_start_as_averages() -> None:
    network = XLSTMTimeForecastNetwork(MODELS[0], FORECAST_SHAPE_CASES[0])
    assert torch.allclose(network.seasonal_linear.weight, torch.full((6, 24), 1.0 / 24))
    assert torch.allclose(network.trend_linear.weight, torch.full((6, 24), 1.0 / 24))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="moving_average_size"):
        XLSTMTimeParameters(moving_average_size=4)
    with pytest.raises(ValueError, match="model_size"):
        XLSTMTimeParameters(model_size=0)
    with pytest.raises(ValueError, match="slstm_num_heads"):
        XLSTMTimeParameters(model_size=6)
