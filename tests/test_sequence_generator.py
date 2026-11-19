from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.models.architectures.recurrent.time_gan import TimeGANParameters, TimeGANPhaseSchedule
from time_series_dl.models.architectures.transformer.diffusion_ts import CosineNoiseSchedule, DiffusionTSParameters
from time_series_dl.models.interfaces.protocols.sequence_generation import SequenceGenerationModel
from time_series_dl.tasks.generation import (
    AutocorrelationDistance,
    DiscriminativeScore,
    GenerationEvaluationSettings,
    GenerationMetrics,
    GeneratorSettings,
    MarginalDistance,
    PredictiveScore,
    SequenceGenerator,
)
from time_series_dl.tasks.generation.generation_step import GenerationStep
from time_series_dl.training import EpochScheduledObjective

LENGTH: int = 16


def make_sines(num_sequences: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = np.arange(LENGTH)
    frequencies = rng.uniform(0.1, 0.3, size=(num_sequences, 1, 2))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(num_sequences, 1, 2))
    return np.sin(frequencies * steps[np.newaxis, :, np.newaxis] + phases) * 2.0 + 5.0


def diffusion_ts() -> DiffusionTSParameters:
    return DiffusionTSParameters(
        model_size=16, num_heads=2, encoder_layers=1, decoder_layers=1, noise_schedule=CosineNoiseSchedule(num_steps=20)
    )


def time_gan() -> TimeGANParameters:
    return TimeGANParameters(
        hidden_size=6, num_layers=2, schedule=TimeGANPhaseSchedule(autoencoder_epochs=1, supervised_epochs=1)
    )


def fast_evaluation() -> GenerationEvaluationSettings:
    post_hoc = TrainingSettings(max_epochs=3, batch_size=32, device=Device.CPU, early_stopping_patience=None)
    return GenerationEvaluationSettings(discriminator_training=post_hoc, predictor_training=post_hoc, num_quantiles=20)


def build(model: SequenceGenerationModel, max_epochs: int = 3, learning_rate: float = 1e-3) -> SequenceGenerator:
    return GeneratorSettings(
        model=model,
        training=TrainingSettings(
            max_epochs=max_epochs,
            batch_size=32,
            learning_rate=learning_rate,
            device=Device.CPU,
            early_stopping_patience=None,
        ),
        window_length=LENGTH,
        window_stride=4,
        evaluation=fast_evaluation(),
    ).build()


@pytest.mark.parametrize("model", [diffusion_ts(), time_gan()])
def test_fit_generate_save_and_load(model: SequenceGenerationModel, tmp_path: Path) -> None:
    generator = build(model)
    real = make_sines(64, seed=0)
    history = generator.fit(real, validation=make_sines(16, seed=1))
    assert history.num_epochs == 3
    samples = generator.generate(40, random_seed=7)
    assert samples.shape == (40, LENGTH, 2)
    assert np.all(np.isfinite(samples))
    np.testing.assert_array_equal(samples, generator.generate(40, random_seed=7))
    assert not np.array_equal(samples, generator.generate(40, random_seed=8))
    path = tmp_path / "generator.pt"
    generator.save(path)
    loaded = SequenceGenerator.load(path, device=Device.CPU)
    np.testing.assert_allclose(loaded.generate(40, random_seed=7), samples, rtol=1e-6)
    assert loaded.history == generator.history
    assert loaded.settings == generator.settings
    metrics = loaded.evaluate(real[:20])
    assert 0.0 <= metrics.discriminative_score <= 0.5
    assert metrics.predictive_score >= 0.0


def test_time_gan_samples_stay_in_the_training_range() -> None:
    generator = build(time_gan())
    real = make_sines(64, seed=0)
    _ = generator.fit(real)
    samples = generator.generate(32, random_seed=0)
    # The recovery network ends with a sigmoid, so samples lie within the min-max scaled training range.
    assert samples.min() >= real.min() - 1e-6
    assert samples.max() <= real.max() + 1e-6


def test_fit_series_cuts_windows() -> None:
    steps = np.arange(200)
    series = np.stack([np.sin(steps / 3.0), np.cos(steps / 5.0)], axis=1)
    generator = build(diffusion_ts(), max_epochs=1)
    _ = generator.fit_series([series, series[:100]], validation_series=series[:50])
    assert generator.generate(3, random_seed=0).shape == (3, LENGTH, 2)
    with pytest.raises(ValueError, match="at least 16 steps"):
        _ = generator.fit_series(series[:10])
    unwindowed = GeneratorSettings(model=diffusion_ts()).build()
    with pytest.raises(ValueError, match="window_length must be set"):
        _ = unwindowed.fit_series(series)


def test_trained_generator_beats_an_untrained_one() -> None:
    real = make_sines(128, seed=0)
    trained = build(diffusion_ts(), max_epochs=30)
    untrained = build(diffusion_ts(), max_epochs=1, learning_rate=1e-12)
    _ = trained.fit(real)
    _ = untrained.fit(real)
    test = make_sines(128, seed=1)
    marginal = MarginalDistance(num_quantiles=50)
    autocorrelation = AutocorrelationDistance(max_lag=None)
    trained_samples = trained.generate(128, random_seed=0)
    untrained_samples = untrained.generate(128, random_seed=0)
    assert marginal.compute(test, trained_samples) < 0.5 * marginal.compute(test, untrained_samples)
    assert autocorrelation.compute(test, trained_samples) < autocorrelation.compute(test, untrained_samples)


def test_generation_step_announces_the_epoch() -> None:
    generator = build(time_gan())
    _ = generator.fit(make_sines(8, seed=0))
    step = GenerationStep(network=generator.network)
    assert isinstance(step, EpochScheduledObjective)
    step.start_epoch(4)
    assert step.epoch == 4
    with pytest.raises(ValueError, match="epoch"):
        step.start_epoch(-1)


def test_errors() -> None:
    generator = build(diffusion_ts())
    with pytest.raises(NotFittedError):
        _ = generator.generate(2)
    with pytest.raises(NotFittedError):
        _ = generator.network
    with pytest.raises(ValueError, match="needs at least 3 epochs"):
        _ = build(time_gan(), max_epochs=2).fit(make_sines(8, seed=0))
    _ = generator.fit(make_sines(8, seed=0))
    with pytest.raises(ValueError, match="num_samples"):
        _ = generator.generate(0)
    with pytest.raises(ValueError, match="length 16 and 2 channels"):
        _ = generator.evaluate(make_sines(8, seed=0)[:, :10])
    with pytest.raises(ValueError, match="window_length"):
        _ = GeneratorSettings(model=diffusion_ts(), window_length=0)


def test_marginal_distance() -> None:
    values = np.random.default_rng(0).normal(size=(30, 8, 2))
    distance = MarginalDistance(num_quantiles=40)
    assert distance.compute(values, values) == 0.0
    assert distance.compute(values, values + 0.25) == pytest.approx(0.25)
    shifted = values.copy()
    shifted[..., 1] += 1.0
    assert distance.compute(values, shifted) == pytest.approx(0.5)


def test_autocorrelation_distance_matches_explicit_computation() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(size=(5, 9, 2))
    synthetic = np.cumsum(rng.normal(size=(7, 9, 2)), axis=1)

    def mean_autocorrelation(values: np.ndarray, num_lags: int) -> np.ndarray:
        centered = values - values.mean(axis=1, keepdims=True)
        variance = (centered**2).sum(axis=1)
        lags = [(centered[:, :-lag] * centered[:, lag:]).sum(axis=1) / variance for lag in range(1, num_lags + 1)]
        return np.stack(lags, axis=1).mean(axis=0)

    for max_lag, num_lags in ((None, 8), (3, 3), (20, 8)):
        expected = np.abs(mean_autocorrelation(real, num_lags) - mean_autocorrelation(synthetic, num_lags)).mean()
        assert AutocorrelationDistance(max_lag).compute(real, synthetic) == pytest.approx(expected)
    constant = np.ones((3, 9, 2))
    assert AutocorrelationDistance(None).compute(constant, constant) == 0.0
    with pytest.raises(ValueError, match="at least 2 steps"):
        _ = AutocorrelationDistance(None).compute(real[:, :1], real[:, :1])


def test_discriminative_score_tells_distinguishable_data_apart() -> None:
    training = TrainingSettings(
        max_epochs=15, batch_size=32, learning_rate=1e-2, device=Device.CPU, early_stopping_patience=None
    )
    score = DiscriminativeScore(training, test_fraction=0.25, split_seed=0)
    real = (make_sines(80, seed=0) - 5.0) / 2.0
    noise = np.random.default_rng(1).uniform(-1.0, 1.0, size=real.shape)
    assert score.compute(real, noise + 3.0) == pytest.approx(0.5)
    similar = score.compute(real, (make_sines(80, seed=1) - 5.0) / 2.0)
    assert similar < 0.2
    assert score.compute(real, (make_sines(80, seed=1) - 5.0) / 2.0) == similar
    with pytest.raises(ValueError, match="at least two"):
        _ = score.compute(real[:1], noise)


def test_predictive_score_prefers_faithful_data() -> None:
    training = TrainingSettings(max_epochs=20, batch_size=16, device=Device.CPU, early_stopping_patience=None)
    score = PredictiveScore(training)
    real = (make_sines(64, seed=0) - 5.0) / 2.0
    faithful = score.compute(real, (make_sines(64, seed=1) - 5.0) / 2.0)
    unfaithful = score.compute(real, np.random.default_rng(1).uniform(-1.0, 1.0, size=real.shape))
    assert faithful < unfaithful
    assert score.compute(real, (make_sines(64, seed=1) - 5.0) / 2.0) == faithful
    univariate = score.compute(real[..., :1], real[..., :1])
    assert np.isfinite(univariate)


def test_metrics_combine_every_measure() -> None:
    real = (make_sines(20, seed=0) - 5.0) / 2.0
    synthetic = (make_sines(20, seed=1) - 5.0) / 2.0
    metrics = GenerationMetrics.compute(real, synthetic, fast_evaluation())
    assert metrics.marginal_distance == MarginalDistance(20).compute(real, synthetic)
    assert metrics.autocorrelation_distance == AutocorrelationDistance(None).compute(real, synthetic)
    with pytest.raises(ValueError, match="test_fraction"):
        _ = GenerationEvaluationSettings(test_fraction=1.0)
