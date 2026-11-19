import math

import numpy as np
import pytest
import torch
from generation_network_checks import check_sequence_generation_network
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.transformer.diffusion_ts import (
    CosineNoiseSchedule,
    DiffusionTSDecomposition,
    DiffusionTSLossKind,
    DiffusionTSParameters,
    DiffusionTSSequenceGenerationNetwork,
)

SHAPE: SequenceShape = SequenceShape(length=12, channels=2)


def small_parameters(
    loss_kind: DiffusionTSLossKind = DiffusionTSLossKind.ABSOLUTE, fourier_weight: float | None = None
) -> DiffusionTSParameters:
    return DiffusionTSParameters(
        model_size=8,
        num_heads=2,
        encoder_layers=1,
        decoder_layers=2,
        feedforward_factor=2,
        noise_schedule=CosineNoiseSchedule(num_steps=6),
        loss_kind=loss_kind,
        fourier_weight=fourier_weight,
    )


def build(
    loss_kind: DiffusionTSLossKind = DiffusionTSLossKind.ABSOLUTE, fourier_weight: float | None = None
) -> DiffusionTSSequenceGenerationNetwork:
    _ = torch.manual_seed(0)
    network = small_parameters(loss_kind, fourier_weight).build_sequence_generation_network(SHAPE)
    assert isinstance(network, DiffusionTSSequenceGenerationNetwork)
    return network.eval()


def cosine_variances(num_steps: int) -> np.ndarray:
    levels = [math.cos((step / num_steps + 0.008) / 1.008 * math.pi / 2.0) ** 2 for step in range(num_steps + 1)]
    return np.array([min(1.0 - levels[step + 1] / levels[step], 0.999) for step in range(num_steps)])


@pytest.mark.parametrize("length", [1, 2, 7, 12])
def test_sequence_generation_network(length: int) -> None:
    network = check_sequence_generation_network(small_parameters(), SequenceShape(length=length, channels=3))
    assert network.minimum_epochs == 1


def test_cosine_schedule_matches_its_formula() -> None:
    variances = CosineNoiseSchedule(num_steps=50).compute_variances().numpy()
    np.testing.assert_allclose(variances, cosine_variances(50), rtol=1e-10)
    assert variances[-1] == pytest.approx(0.999)
    with pytest.raises(ValueError, match="max_variance"):
        _ = CosineNoiseSchedule(max_variance=1.0)


@pytest.mark.parametrize("loss_kind", [DiffusionTSLossKind.ABSOLUTE, DiffusionTSLossKind.SQUARED])
def test_denoising_loss_matches_explicit_computation(loss_kind: DiffusionTSLossKind) -> None:
    network = build(loss_kind, fourier_weight=0.7)
    estimate = torch.randn(3, SHAPE.length, SHAPE.channels, dtype=torch.float64)
    clean = torch.randn(3, SHAPE.length, SHAPE.channels, dtype=torch.float64)
    steps = torch.tensor([0, 5, 2])
    loss = network.compute_denoising_loss(estimate, clean, steps).item()

    def error(difference: np.ndarray) -> np.ndarray:
        return np.abs(difference) if loss_kind == DiffusionTSLossKind.ABSOLUTE else difference**2

    length = SHAPE.length
    frequencies = np.arange(length)
    transform = np.exp(-2j * np.pi * np.outer(frequencies, frequencies) / length) / length
    gap = np.einsum("kt,btc->bkc", transform, estimate.numpy() - clean.numpy())
    total = error(estimate.numpy() - clean.numpy()) + 0.7 * (error(gap.real) + error(gap.imag))
    betas = cosine_variances(6)
    alpha_bars = np.cumprod(1.0 - betas)
    weights = np.sqrt(1.0 - betas) * np.sqrt(1.0 - alpha_bars) / betas / 100.0
    expected = np.mean(total.reshape(3, -1).mean(axis=1) * weights[steps.numpy()])
    assert loss == pytest.approx(expected, rel=1e-5)


def test_default_fourier_weight_depends_on_the_length() -> None:
    assert build().fourier_weight == pytest.approx(math.sqrt(SHAPE.length) / 5.0)


def test_loss_is_the_denoising_loss_of_a_random_step() -> None:
    network = build()
    values = torch.rand(4, SHAPE.length, SHAPE.channels)
    _ = torch.manual_seed(5)
    loss = network.compute_loss(values, 0)
    _ = torch.manual_seed(5)
    noised = network.process.corrupt(values)
    expected = network.compute_denoising_loss(network.denoiser(noised.noisy, noised.steps), values, noised.steps)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_decomposition_is_interpretable() -> None:
    network = build()
    noisy = torch.randn(3, SHAPE.length, SHAPE.channels)
    steps = torch.tensor([0, 3, 5])
    with torch.no_grad():
        parts: DiffusionTSDecomposition = network.denoiser.decompose(noisy, steps)
        estimate: Tensor = network.denoiser(noisy, steps)
    assert torch.allclose(parts.estimate, estimate)
    assert torch.allclose(parts.residual.mean(dim=1), torch.zeros(3, SHAPE.channels), atol=1e-6)
    # The trend is a cubic polynomial of the normalized time in every sequence and channel.
    times = np.arange(1, SHAPE.length + 1) / (SHAPE.length + 1)
    basis = np.stack([times**power for power in range(4)], axis=1)
    trends = parts.trend.double().numpy().transpose(1, 0, 2).reshape(SHAPE.length, -1)
    coefficients = np.linalg.lstsq(basis, trends, rcond=None)[0]
    np.testing.assert_allclose(basis @ coefficients, trends, atol=1e-5)


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="model_size"):
        _ = DiffusionTSParameters(model_size=6, num_heads=4)
    with pytest.raises(ValueError, match="model_size must be even"):
        _ = DiffusionTSParameters(model_size=7, num_heads=1)
    with pytest.raises(ValueError, match="dropout"):
        _ = DiffusionTSParameters(dropout=1.0)
    with pytest.raises(ValueError, match="fourier_weight"):
        _ = DiffusionTSParameters(fourier_weight=-1.0)
    with pytest.raises(ValueError, match="trend_degree"):
        _ = DiffusionTSParameters(trend_degree=0)
