from typing import cast

import pytest
import torch
import torch.nn.functional as functional
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.mlp.frets import FreTSFrequencyLayer, FreTSParameters

MODELS: list[FreTSParameters] = [
    FreTSParameters(embed_size=8, hidden_size=16),
    FreTSParameters(embed_size=4, hidden_size=8, sparsity_threshold=0.0, is_channel_mixing=False),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FreTSParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_frequency_layer_matches_complex_reference() -> None:
    _ = torch.manual_seed(0)
    layer: FreTSFrequencyLayer = FreTSFrequencyLayer(embed_size=4, sparsity_threshold=0.01, scale=0.5)
    tokens: Tensor = torch.randn(2, 3, 10, 4)
    spectrum: Tensor = cast(Tensor, torch.fft.rfft(tokens, dim=2, norm="ortho"))
    weight: Tensor = torch.view_as_complex(layer.weight.detach().contiguous())
    bias: Tensor = torch.view_as_complex(layer.bias.detach().contiguous())
    linear: Tensor = torch.einsum("bijd,de->bije", spectrum, weight) + bias
    real: Tensor = functional.softshrink(functional.relu(linear.real), 0.01)
    imaginary: Tensor = functional.softshrink(functional.relu(linear.imag), 0.01)
    expected: Tensor = cast(Tensor, torch.fft.irfft(torch.complex(real, imaginary), n=10, dim=2, norm="ortho"))
    with torch.no_grad():
        assert torch.allclose(layer(tokens, 2), expected, atol=1e-6)


@pytest.mark.parametrize("is_channel_mixing", [True, False])
def test_channel_learner_controls_cross_channel_dependence(is_channel_mixing: bool) -> None:
    _ = torch.manual_seed(0)
    parameters: FreTSParameters = FreTSParameters(
        embed_size=8, hidden_size=16, weight_scale=0.5, is_channel_mixing=is_channel_mixing
    )
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(0,))
    network = parameters.build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += torch.randn(2, 12)
    with torch.no_grad():
        is_unchanged: bool = torch.allclose(network(past), network(changed))
    assert is_unchanged != is_channel_mixing


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="embed_size"):
        FreTSParameters(embed_size=0)
    with pytest.raises(ValueError, match="hidden_size"):
        FreTSParameters(hidden_size=0)
    with pytest.raises(ValueError, match="sparsity_threshold"):
        FreTSParameters(sparsity_threshold=-0.1)
    with pytest.raises(ValueError, match="weight_scale"):
        FreTSParameters(weight_scale=0.0)
