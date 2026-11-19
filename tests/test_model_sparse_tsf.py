import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.linear.sparse_tsf import SparseTSFParameters

MODELS: list[SparseTSFParameters] = [SparseTSFParameters(), SparseTSFParameters(period=4)]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: SparseTSFParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_forecast_matches_reference_computation() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=2, horizon=8, target_channels=(1,))
    network = SparseTSFParameters(period=4).build_forecast_network(shape).eval()
    kernel: Tensor = network.get_parameter("aggregation.weight").detach()[0, 0]
    weight: Tensor = network.get_parameter("linear.weight").detach()
    past: Tensor = torch.randn(3, 12, 2)
    series: Tensor = past[:, :, 1]
    padded: Tensor = torch.nn.functional.pad(series, (2, 2))
    aggregated: Tensor = series + torch.stack([padded[:, step : step + 5] @ kernel for step in range(12)], dim=1)
    periods: Tensor = aggregated.reshape(3, 3, 4)
    expected: Tensor = torch.einsum("bnp,mn->bmp", periods, weight).reshape(3, 8)
    with torch.no_grad():
        assert torch.allclose(network(past)[:, :, 0, 0], expected, atol=1e-5)


def test_phases_share_the_linear_map() -> None:
    shape: ForecastShape = ForecastShape(input_length=6, input_channels=1, horizon=4, target_channels=(0,))
    network = SparseTSFParameters(period=2).build_forecast_network(shape).eval()
    with torch.no_grad():
        _ = network.get_parameter("aggregation.weight").zero_()
        past: Tensor = torch.tensor([[1.0, 5.0, 2.0, 6.0, 3.0, 7.0]]).unsqueeze(-1)
        swapped: Tensor = torch.tensor([[5.0, 1.0, 6.0, 2.0, 7.0, 3.0]]).unsqueeze(-1)
        outputs: Tensor = network(past)[0, :, 0, 0]
        swapped_outputs: Tensor = network(swapped)[0, :, 0, 0]
    assert torch.allclose(outputs.reshape(2, 2).flip(1), swapped_outputs.reshape(2, 2))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="period"):
        SparseTSFParameters(period=0)
