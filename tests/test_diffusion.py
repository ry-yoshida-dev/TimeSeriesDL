import math

import pytest
import torch

from time_series_dl.models.components.diffusion import (
    DiffusionProcess,
    DiffusionStepEmbedding,
    NoisedSample,
    NoiseSchedule,
    NoiseScheduleKind,
)


def test_linear_and_quadratic_schedules() -> None:
    linear = NoiseSchedule(kind=NoiseScheduleKind.LINEAR, num_steps=5, first_variance=0.1, last_variance=0.5)
    assert linear.compute_variances().tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    quadratic = NoiseSchedule(kind=NoiseScheduleKind.QUADRATIC, num_steps=3, first_variance=0.01, last_variance=0.25)
    # Square roots evenly spaced from 0.1 to 0.5.
    assert quadratic.compute_variances().tolist() == pytest.approx([0.01, 0.09, 0.25])


def test_process_buffers_follow_the_schedule() -> None:
    process = NoiseSchedule(num_steps=4, first_variance=0.1, last_variance=0.4).build_process()
    betas = [0.1, 0.2, 0.3, 0.4]
    alpha_bars = [0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.7, 0.9 * 0.8 * 0.7 * 0.6]
    assert process.num_steps == 4
    assert process.alpha_bars.tolist() == pytest.approx(alpha_bars)
    assert process.signal_scales.tolist() == pytest.approx([math.sqrt(value) for value in alpha_bars])
    assert process.noise_scales.tolist() == pytest.approx([math.sqrt(1.0 - value) for value in alpha_bars])
    posterior = [0.0] + [(1.0 - alpha_bars[step - 1]) / (1.0 - alpha_bars[step]) * betas[step] for step in range(1, 4)]
    assert process.posterior_deviations.tolist() == pytest.approx([math.sqrt(value) for value in posterior])
    assert not dict(process.state_dict())


def test_diffuse_matches_closed_form() -> None:
    process = NoiseSchedule(num_steps=10).build_process()
    clean = torch.randn(3, 2, 5)
    noise = torch.randn(3, 2, 5)
    step_values = [0, 4, 9]
    noisy = process.diffuse(clean, torch.tensor(step_values), noise)
    for index, step in enumerate(step_values):
        alpha_bar = float(process.alpha_bars[step])
        expected = math.sqrt(alpha_bar) * clean[index] + math.sqrt(1.0 - alpha_bar) * noise[index]
        assert torch.allclose(noisy[index], expected, atol=1e-6)
    with pytest.raises(ValueError, match="steps"):
        _ = process.diffuse(clean, torch.tensor([0, 1, 10]), noise)


def test_corrupt_draws_steps_in_range_reproducibly() -> None:
    process = NoiseSchedule(num_steps=7).build_process()
    clean = torch.zeros(50, 3)
    first: NoisedSample = process.corrupt(clean, torch.Generator().manual_seed(0))
    second: NoisedSample = process.corrupt(clean, torch.Generator().manual_seed(0))
    assert first.steps.shape == (50,)
    assert int(first.steps.min()) >= 0 and int(first.steps.max()) < 7
    assert torch.equal(first.noisy, second.noisy) and torch.equal(first.steps, second.steps)
    assert torch.allclose(first.noisy, process.noise_scales[first.steps].unsqueeze(-1) * first.noise)


def test_sampler_shapes_and_determinism() -> None:
    process = NoiseSchedule(num_steps=6).build_process()
    calls: list[int] = []

    def predict_noise(noisy: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        calls.append(int(steps[0]))
        assert steps.shape == (4,)
        return 0.5 * noisy

    first = process.sample(predict_noise, (4, 2, 3), torch.Generator().manual_seed(0))
    assert first.shape == (4, 2, 3)
    assert calls == [5, 4, 3, 2, 1, 0]
    second = process.sample(predict_noise, (4, 2, 3), torch.Generator().manual_seed(0))
    third = process.sample(predict_noise, (4, 2, 3), torch.Generator().manual_seed(1))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    with pytest.raises(ValueError, match="positive"):
        _ = process.sample(predict_noise, (0, 2))


def test_sampler_with_the_true_noise_recovers_a_point_mass() -> None:
    # With data that is always x0, the optimal noise prediction is (x_n - sqrt(ab) x0) / sqrt(1 - ab).
    process = NoiseSchedule(num_steps=50, last_variance=0.5).build_process()
    target = torch.tensor([1.5, -2.0])

    def predict_noise(noisy: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        signal = process.signal_scales[steps].unsqueeze(-1)
        spread = process.noise_scales[steps].unsqueeze(-1)
        return (noisy - signal * target) / spread

    samples = process.sample(predict_noise, (64, 2), torch.Generator().manual_seed(0))
    assert torch.allclose(samples, target.expand(64, 2), atol=1e-3)


def test_schedule_and_process_validation() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        _ = NoiseSchedule(num_steps=0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        _ = NoiseSchedule(last_variance=1.0)
    with pytest.raises(ValueError, match="exceed"):
        _ = NoiseSchedule(first_variance=0.2, last_variance=0.1)
    with pytest.raises(ValueError, match="vector"):
        _ = DiffusionProcess(torch.zeros(0))


def test_step_embedding() -> None:
    embedding = DiffusionStepEmbedding(8, 5)
    assert embedding(torch.tensor([0, 3, 99])).shape == (3, 5)
    with pytest.raises(ValueError, match="even"):
        _ = DiffusionStepEmbedding(7, 5)
