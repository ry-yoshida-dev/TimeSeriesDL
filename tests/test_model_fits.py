from typing import cast

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape
from time_series_dl.models.architectures.linear.fits import FITSFrequencyInterpolation, FITSParameters

MODELS: list[FITSParameters] = [
    FITSParameters(),
    FITSParameters(base_period=4, harmonic_order=1, is_individual=True),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FITSParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: FITSParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


def test_cutoff_follows_official_formula() -> None:
    assert FITSParameters(base_period=24, harmonic_order=6).compute_cutoff_frequency(720) == 196


def test_interpolation_matches_reference_computation() -> None:
    _ = torch.manual_seed(0)
    interpolation: FITSFrequencyInterpolation = FITSFrequencyInterpolation(
        input_length=16, output_length=24, cutoff_frequency=4, channels=2, is_individual=False, num_outputs=1
    )
    values: Tensor = torch.randn(3, 16, 2)
    weight: Tensor = torch.view_as_complex(interpolation.weight.detach())[0]
    bias: Tensor = torch.view_as_complex(interpolation.bias.detach())[0]
    spectrum: Tensor = cast(Tensor, torch.fft.rfft(values, dim=1))[:, :4]
    mixed: Tensor = torch.einsum("bfc,fg->bgc", spectrum, weight) + bias[None, :, None]
    full: Tensor = torch.zeros(3, 13, 2, dtype=mixed.dtype)
    full[:, :6] = mixed
    expected: Tensor = cast(Tensor, torch.fft.irfft(full, n=24, dim=1)) * 1.5
    with torch.no_grad():
        assert torch.allclose(interpolation(values)[..., 0], expected, atol=1e-5)


def test_low_pass_filter_ignores_high_frequencies() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=32, input_channels=1, horizon=8, target_channels=(0,))
    network = FITSParameters(base_period=32, harmonic_order=1).build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 32, 1)
    noise: Tensor = torch.cos(torch.pi * torch.arange(32.0))[None, :, None]
    with torch.no_grad():
        assert torch.allclose(network(past), network(past + noise), atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="base_period"):
        FITSParameters(base_period=0)
    with pytest.raises(ValueError, match="harmonic_order"):
        FITSParameters(harmonic_order=0)
    with pytest.raises(ValueError, match="output_length"):
        FITSFrequencyInterpolation(
            input_length=8, output_length=4, cutoff_frequency=2, channels=1, is_individual=False, num_outputs=1
        )
