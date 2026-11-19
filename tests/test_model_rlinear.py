import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape
from time_series_dl.models.architectures.linear.rlinear import RLinearParameters

MODELS: list[RLinearParameters] = [
    RLinearParameters(),
    RLinearParameters(is_individual=True, dropout=0.0, is_affine=False),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: RLinearParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: RLinearParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


def test_forecast_matches_reference_computation() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(2, 0))
    network = RLinearParameters(dropout=0.0).build_forecast_network(shape).eval()
    with torch.no_grad():
        for parameter in network.parameters():
            _ = parameter.uniform_(0.5, 1.5)
    weight: Tensor = network.get_parameter("normalization.weight")
    bias: Tensor = network.get_parameter("normalization.bias")
    linear_weight: Tensor = network.get_parameter("linear.weight")[0]
    linear_bias: Tensor = network.get_parameter("linear.bias")[0]
    past: Tensor = torch.randn(2, 12, 3) * 3.0 + 5.0
    targets: Tensor = past[:, :, [2, 0]]
    mean: Tensor = targets.mean(dim=1, keepdim=True)
    deviation: Tensor = torch.sqrt(targets.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
    normalized: Tensor = (targets - mean) / deviation * weight + bias
    mapped: Tensor = torch.einsum("blc,lh->bhc", normalized, linear_weight) + linear_bias[None, :, None]
    expected: Tensor = (mapped - bias) / (weight + 1e-10) * deviation + mean
    with torch.no_grad():
        assert torch.allclose(network(past)[..., 0], expected, atol=1e-4)


def test_only_level_parameters_are_denormalized() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(
        input_length=10, input_channels=1, horizon=3, target_channels=(0,), num_parameters=2
    )
    network = RLinearParameters(dropout=0.0).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 10, 1)
    with torch.no_grad():
        original: Tensor = network(past)
        shifted: Tensor = network(past + 4.0)
    assert torch.allclose(shifted[..., 0], original[..., 0] + 4.0, atol=1e-4)
    assert torch.allclose(shifted[..., 1], original[..., 1], atol=1e-4)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="dropout"):
        RLinearParameters(dropout=1.0)
