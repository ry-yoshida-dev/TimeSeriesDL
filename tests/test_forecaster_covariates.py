import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.data import CovariateMismatchError, SeriesCovariates, WindowCovariates
from time_series_dl.models import (
    DeepARParameters,
    ForecastingModel,
    MLPParameters,
    TemporalFusionTransformerParameters,
    TiDEParameters,
)
from time_series_dl.models.architectures.mlp.tide import TiDECovariateForecastNetwork
from time_series_dl.models.components.covariates import CovariateAdapterNetwork
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow, QuantileObjective

WINDOW: ForecastWindow = ForecastWindow(input_length=24, horizon=6)


def make_promotion_series(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A seasonal series lifted by 2 on random promotion days, and the promotion indicator."""
    rng = np.random.default_rng(seed)
    steps = np.arange(length)
    promotion = (rng.random(length) < 0.3).astype(np.float64)
    values = np.sin(steps / 10.0) + 2.0 * promotion + rng.normal(scale=0.05, size=length)
    return values[:, np.newaxis], promotion[:, np.newaxis]


@pytest.mark.parametrize(
    "model",
    [
        MLPParameters(hidden_sizes=(32,)),
        TiDEParameters(hidden_size=32, temporal_decoder_hidden_size=16, dropout=0.0),
        DeepARParameters(hidden_size=16),
        TemporalFusionTransformerParameters(hidden_size=16, num_heads=2, dropout=0.0),
    ],
    ids=["adapter", "tide", "deep_ar", "temporal_fusion_transformer"],
)
def test_known_future_covariate_carries_the_unpredictable_signal(model: ForecastingModel) -> None:
    series, promotion = make_promotion_series(600, seed=0)
    training = TrainingSettings(
        max_epochs=15, batch_size=32, learning_rate=3e-3, device=Device.CPU, early_stopping_patience=None, random_seed=0
    )
    without = ForecasterSettings(model=model, window=WINDOW, training=training).build()
    _ = without.fit(series[:450])
    with_covariates = ForecasterSettings(model=model, window=WINDOW, training=training).build()
    _ = with_covariates.fit(series[:450], covariates=SeriesCovariates(known=promotion[:450]))
    blind_error: float = without.evaluate(series[450:]).mae
    informed_error: float = with_covariates.evaluate(series[450:], SeriesCovariates(known=promotion[450:])).mae
    assert blind_error > 0.6
    assert informed_error < 0.35


def test_forecast_reads_known_covariates_over_the_horizon(fast_training: TrainingSettings) -> None:
    series, promotion = make_promotion_series(120, seed=1)
    forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(8,)), window=WINDOW, training=fast_training
    ).build()
    _ = forecaster.fit(series[:100], covariates=SeriesCovariates(known=promotion[:100]))
    assert isinstance(forecaster.network, CovariateAdapterNetwork)
    covariates = SeriesCovariates(known=promotion[: 100 + 6])
    prediction = forecaster.forecast(series[:100], covariates)
    assert prediction.point.shape == (1, 6, 1)
    manual = forecaster.predict(series[np.newaxis, 76:100], WindowCovariates(known=promotion[np.newaxis, 76:106]))
    np.testing.assert_allclose(prediction.point, manual.point)
    with pytest.raises(CovariateMismatchError, match="106 steps"):
        _ = forecaster.forecast(series[:100], SeriesCovariates(known=promotion[:100]))
    with pytest.raises(CovariateMismatchError, match="fitted with covariates"):
        _ = forecaster.forecast(series[:100])


def test_evaluate_aligns_covariates_with_the_windows(fast_training: TrainingSettings) -> None:
    series, promotion = make_promotion_series(150, seed=2)
    forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        window=ForecastWindow(input_length=24, horizon=6, stride=5),
        training=fast_training,
    ).build()
    _ = forecaster.fit(series[:100], covariates=SeriesCovariates(known=promotion[:100]))
    metrics = forecaster.evaluate(series[100:], SeriesCovariates(known=promotion[100:]))
    starts = range(100, 150 - 30 + 1, 5)
    prediction = forecaster.predict(
        np.stack([series[start : start + 24] for start in starts]),
        WindowCovariates(known=np.stack([promotion[start : start + 30] for start in starts])),
    )
    truth = np.stack([series[start + 24 : start + 30] for start in starts])
    assert metrics.num_windows == len(starts)
    assert metrics.mae == pytest.approx(float(np.mean(np.abs(prediction.point - truth))), rel=1e-5)


def test_static_covariates_of_several_series(fast_training: TrainingSettings) -> None:
    first, first_promotion = make_promotion_series(80, seed=3)
    second, second_promotion = make_promotion_series(90, seed=4)
    covariates = [
        SeriesCovariates(known=first_promotion, static=np.array([1.0, 5.0]), static_categories=np.array([0, 2])),
        SeriesCovariates(known=second_promotion, static=np.array([3.0, 2.0]), static_categories=np.array([1, 0])),
    ]
    forecaster = ForecasterSettings(
        model=TiDEParameters(hidden_size=8, temporal_decoder_hidden_size=4),
        window=WINDOW,
        objective=QuantileObjective((0.1, 0.5, 0.9)),
        training=fast_training,
    ).build()
    _ = forecaster.fit([first, second], covariates=covariates)
    network = forecaster.network
    assert isinstance(network, TiDECovariateForecastNetwork)
    # Two numeric features, the location and log scale of the target, and embeddings of 2 and 3 categories.
    assert network.static_embedding.output_size == 2 + 2 + 1 + 2
    assert forecaster.evaluate([first, second], covariates).num_windows == (80 - 29) + (90 - 29)
    unseen = SeriesCovariates(known=first_promotion, static=np.array([1.0, 5.0]), static_categories=np.array([2, 0]))
    with pytest.raises(CovariateMismatchError, match="not seen in training"):
        _ = forecaster.evaluate(first, unseen)
    with pytest.raises(CovariateMismatchError, match="dimensions"):
        _ = forecaster.evaluate(first, SeriesCovariates(known=first_promotion))


def test_covariates_must_match_how_the_forecaster_was_fitted(fast_training: TrainingSettings) -> None:
    series, promotion = make_promotion_series(100, seed=5)
    plain = ForecasterSettings(model=MLPParameters(hidden_sizes=(8,)), window=WINDOW, training=fast_training).build()
    with pytest.raises(NotFittedError):
        _ = plain.forecast(series[:94], SeriesCovariates(known=promotion))
    _ = plain.fit(series)
    with pytest.raises(CovariateMismatchError, match="fitted without covariates"):
        _ = plain.evaluate(series, SeriesCovariates(known=promotion))
    covariate_forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(8,)), window=WINDOW, training=fast_training
    ).build()
    with pytest.raises(CovariateMismatchError, match="validation_covariates must be given"):
        _ = covariate_forecaster.fit(series[:70], series[70:], covariates=SeriesCovariates(known=promotion[:70]))
    with pytest.raises(CovariateMismatchError, match="without validation_series"):
        _ = covariate_forecaster.fit(series, validation_covariates=SeriesCovariates(known=promotion))
    with pytest.raises(CovariateMismatchError, match="70 steps"):
        _ = covariate_forecaster.fit(series[:70], covariates=SeriesCovariates(known=promotion[:60]))
    history = covariate_forecaster.fit(
        series[:70],
        series[70:],
        covariates=SeriesCovariates(known=promotion[:70]),
        validation_covariates=SeriesCovariates(known=promotion[70:]),
    )
    assert len(history.validation_losses) == fast_training.max_epochs
    windows = np.stack([series[:24], series[10:34]])
    with pytest.raises(CovariateMismatchError, match="2 windows"):
        _ = covariate_forecaster.predict(windows, WindowCovariates(known=promotion[np.newaxis, :30]))
    with pytest.raises(CovariateMismatchError, match="cover 30 steps"):
        _ = covariate_forecaster.predict(windows, WindowCovariates(known=np.stack([promotion[:24], promotion[10:34]])))
    assert issubclass(CovariateMismatchError, ValueError)
