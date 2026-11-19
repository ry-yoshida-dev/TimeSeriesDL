import math

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.mlp.koopa import (
    KoopaDynamicModeDecomposition,
    KoopaFourierFilter,
    KoopaParameters,
)

FEASIBLE_SHAPES: list[ForecastShape] = [shape for shape in FORECAST_SHAPE_CASES if shape.input_length > shape.horizon]

MODELS: list[KoopaParameters] = [
    KoopaParameters(num_blocks=2, dynamic_size=8, hidden_size=16),
    KoopaParameters(num_blocks=1, dynamic_size=4, hidden_size=8, num_layers=3, invariant_fraction=0.0, dropout=0.0),
]


@pytest.mark.parametrize("shape", FEASIBLE_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: KoopaParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_filter_fixes_mask_after_first_training_batch() -> None:
    steps: Tensor = torch.arange(32, dtype=torch.float32)
    first: Tensor = torch.sin(2 * torch.pi * 4 * steps / 32)[None, :, None].expand(2, -1, 3)
    second: Tensor = torch.sin(2 * torch.pi * 9 * steps / 32)[None, :, None].expand(2, -1, 3)
    filter_layer: KoopaFourierFilter = KoopaFourierFilter(length=32, invariant_fraction=0.1)
    _ = filter_layer.eval()
    _ = filter_layer(first)
    assert not bool(filter_layer.is_initialized)
    _ = filter_layer.train()
    variant: Tensor
    invariant: Tensor
    variant, invariant = filter_layer(first)
    assert bool(filter_layer.is_initialized)
    assert float(filter_layer.invariant_mask[4]) == 1.0
    assert torch.allclose(invariant, first, atol=1e-5)
    assert torch.allclose(variant, torch.zeros_like(first), atol=1e-5)
    variant, invariant = filter_layer(second)
    assert torch.allclose(variant, second, atol=1e-5)
    assert torch.allclose(invariant, torch.zeros_like(second), atol=1e-5)


def test_dynamic_mode_decomposition_recovers_linear_system() -> None:
    _ = torch.manual_seed(0)
    operator: Tensor = 0.95 * torch.tensor([[math.cos(0.3), math.sin(0.3)], [-math.sin(0.3), math.cos(0.3)]])
    states: list[Tensor] = [torch.randn(1, 1, 2)]
    for _ in range(9):
        states.append(states[-1] @ operator)
    sequence: Tensor = torch.cat(states, dim=1)
    future: list[Tensor] = [sequence[:, -1:] @ operator]
    for _ in range(2):
        future.append(future[-1] @ operator)
    reconstruction: Tensor
    predictions: Tensor
    reconstruction, predictions = KoopaDynamicModeDecomposition()(sequence, 3)
    assert torch.allclose(reconstruction, sequence, atol=1e-4)
    assert torch.allclose(predictions, torch.cat(future, dim=1), atol=1e-4)


def test_each_block_forecast_is_finite_with_padding() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=2, horizon=4, target_channels=(1,))
    network = MODELS[0].build_forecast_network(shape).eval()
    with torch.no_grad():
        outputs: Tensor = network(torch.randn(2, 10, 2))
    assert outputs.shape == (2, 4, 1, 1)
    assert bool(torch.isfinite(outputs).all())


def test_rejects_input_not_longer_than_horizon() -> None:
    shape: ForecastShape = ForecastShape(input_length=6, input_channels=1, horizon=6, target_channels=(0,))
    with pytest.raises(ValueError, match="longer than the horizon"):
        KoopaParameters().build_forecast_network(shape)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        KoopaParameters(num_blocks=0)
    with pytest.raises(ValueError, match="dynamic_size"):
        KoopaParameters(dynamic_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        KoopaParameters(num_layers=0)
    with pytest.raises(ValueError, match="invariant_fraction"):
        KoopaParameters(invariant_fraction=1.5)
    with pytest.raises(ValueError, match="dropout"):
        KoopaParameters(dropout=1.0)
