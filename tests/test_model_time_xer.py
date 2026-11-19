import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape
from time_series_dl.models.architectures.transformer.time_xer import (
    TimeXerEndogenousEmbedding,
    TimeXerParameters,
)

MODELS: list[TimeXerParameters] = [
    TimeXerParameters(patch_length=4, model_size=8, num_heads=2, num_layers=2, feedforward_size=16),
    TimeXerParameters(
        patch_length=5, model_size=6, num_heads=3, num_layers=1, feedforward_size=8, activation=Activation.RELU
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TimeXerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize(("input_length", "num_patches"), [(3, 1), (16, 4), (17, 5)])
def test_endogenous_series_are_padded_to_whole_patches(input_length: int, num_patches: int) -> None:
    embedding = TimeXerEndogenousEmbedding(MODELS[0], input_length, num_variates=2)
    assert embedding.num_patches == num_patches
    assert embedding(torch.randn(3, 2, input_length)).shape == (6, num_patches + 1, 8)


def test_forecast_ignores_the_order_of_exogenous_channels() -> None:
    _ = torch.manual_seed(0)
    shape = ForecastShape(input_length=12, input_channels=4, horizon=3, target_channels=(1,))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 4)
    with torch.no_grad():
        original: Tensor = network(past)
        permuted: Tensor = network(past[..., [3, 1, 0, 2]])
    assert torch.allclose(original, permuted, atol=1e-5)


def test_exogenous_channels_influence_the_forecast() -> None:
    _ = torch.manual_seed(0)
    shape = ForecastShape(input_length=12, input_channels=3, horizon=3, target_channels=(0,))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[..., 2] += 1.0
    with torch.no_grad():
        assert not torch.allclose(network(past), network(changed))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="patch_length"):
        TimeXerParameters(patch_length=0)
    with pytest.raises(ValueError, match="num_heads"):
        TimeXerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        TimeXerParameters(dropout=-0.1)
