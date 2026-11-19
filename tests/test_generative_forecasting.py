from pathlib import Path

import numpy as np
import pytest

from time_series_dl import TrainingSettings
from time_series_dl.models import MLPParameters, ModelArchitecture
from time_series_dl.models.architectures.recurrent.time_grad import TimeGradParameters
from time_series_dl.models.architectures.transformer.csdi import CSDIParameters
from time_series_dl.models.components.diffusion import NoiseSchedule
from time_series_dl.models.interfaces.generative_forecast_network import GenerativeForecastNetwork
from time_series_dl.models.interfaces.protocols.generative_forecasting import GenerativeForecastingModel
from time_series_dl.tasks.forecasting import (
    CovariateMismatchError,
    Forecaster,
    ForecasterSettings,
    ForecastWindow,
    GaussianObjective,
    SampleObjective,
    SeriesCovariates,
)

GENERATIVE_MODELS: list[GenerativeForecastingModel] = [
    TimeGradParameters(
        hidden_size=8,
        conditioning_size=8,
        noise_schedule=NoiseSchedule(num_steps=4),
        residual_layers=2,
        residual_channels=4,
        step_feature_size=8,
        step_projection_size=8,
    ),
    CSDIParameters(
        channels=8,
        num_layers=1,
        num_heads=2,
        feedforward_size=8,
        step_feature_size=8,
        time_embedding_size=8,
        feature_embedding_size=4,
        noise_schedule=NoiseSchedule(num_steps=4),
    ),
]


def _is_registered(model: GenerativeForecastingModel) -> bool:
    return type(model).__name__ in {architecture.parameters_type.__name__ for architecture in ModelArchitecture}


def _build(model: GenerativeForecastingModel, training: TrainingSettings) -> Forecaster:
    return ForecasterSettings(
        model=model,
        window=ForecastWindow(input_length=16, horizon=4, stride=4),
        objective=SampleObjective(num_samples=6, prediction_quantiles=(0.1, 0.5, 0.9)),
        training=training,
        target_channels=(1, 0),
    ).build()


@pytest.mark.parametrize("model", GENERATIVE_MODELS)
def test_generative_forecaster_returns_statistics_of_samples(
    model: GenerativeForecastingModel, fast_training: TrainingSettings, sine_series: np.ndarray
) -> None:
    forecaster = _build(model, fast_training)
    history = forecaster.fit(sine_series[:200], sine_series[200:300])
    assert history.num_epochs == 3
    assert isinstance(forecaster.network, GenerativeForecastNetwork)
    prediction = forecaster.forecast(sine_series, num_samples=9)
    assert prediction.samples is not None
    assert prediction.samples.shape == (1, 9, 4, 2)
    assert np.allclose(prediction.point, np.median(prediction.samples, axis=1))
    quantiles = np.moveaxis(np.quantile(prediction.samples, [0.1, 0.5, 0.9], axis=1), 0, -1)
    assert np.allclose(prediction.quantiles, quantiles)
    # Predictions are reproducible, and the default draws the objective's number of samples without returning them.
    repeated = forecaster.forecast(sine_series, num_samples=9)
    assert repeated.samples is not None and np.array_equal(prediction.samples, repeated.samples)
    assert forecaster.forecast(sine_series).samples is None
    metrics = forecaster.evaluate(sine_series[300:], num_samples=5)
    assert metrics.crps is not None and metrics.crps > 0.0
    assert metrics.interval_coverage is not None and set(metrics.interval_coverage) == {0.8}


@pytest.mark.parametrize("model", GENERATIVE_MODELS)
def test_generative_forecaster_round_trips(
    model: GenerativeForecastingModel, fast_training: TrainingSettings, sine_series: np.ndarray, tmp_path: Path
) -> None:
    if not _is_registered(model):
        pytest.skip(f"{type(model).__name__} is not a ModelArchitecture yet, so its settings cannot be saved.")
    forecaster = _build(model, fast_training)
    _ = forecaster.fit(sine_series[:200])
    path = tmp_path / "generative.pt"
    forecaster.save(path)
    loaded = Forecaster.load(path)
    assert loaded.settings == forecaster.settings
    original = forecaster.forecast(sine_series, num_samples=4)
    restored = loaded.forecast(sine_series, num_samples=4)
    assert original.samples is not None and restored.samples is not None
    assert np.allclose(original.samples, restored.samples)


def test_generative_models_need_sample_objective_and_no_covariates(
    fast_training: TrainingSettings, sine_series: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="SampleObjective"):
        _ = ForecasterSettings(model=GENERATIVE_MODELS[0], window=ForecastWindow(16, 4), objective=GaussianObjective())
    forecaster = _build(GENERATIVE_MODELS[1], fast_training)
    covariates = SeriesCovariates(known=np.zeros((sine_series.shape[0], 1)))
    with pytest.raises(CovariateMismatchError, match="generative"):
        _ = forecaster.fit(sine_series, covariates=covariates)


def test_sample_objective_trains_any_network_with_the_crps(
    fast_training: TrainingSettings, sine_series: np.ndarray
) -> None:
    forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(16,)),
        window=ForecastWindow(input_length=16, horizon=4, stride=2),
        objective=SampleObjective(num_samples=8),
        training=fast_training,
    ).build()
    history = forecaster.fit(sine_series)
    assert history.training_losses[-1] < history.training_losses[0]
    prediction = forecaster.forecast(sine_series, num_samples=20)
    assert prediction.samples is not None and prediction.samples.shape == (1, 20, 4, 3)
