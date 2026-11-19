import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import DLinearParameters, MLPParameters, RNNParameters
from time_series_dl.tasks.forecasting import (
    ForecasterSettings,
    ForecastWindow,
    GaussianObjective,
    LaplaceObjective,
    PointObjective,
    QuantileObjective,
    StudentTObjective,
)
from time_series_dl.tasks.forecasting.objective import ForecastObjective


@pytest.mark.parametrize(
    "objective",
    [
        PointObjective(),
        QuantileObjective((0.1, 0.5, 0.9)),
        GaussianObjective(),
        StudentTObjective(),
        LaplaceObjective(),
    ],
)
@pytest.mark.parametrize("is_instance_normalized", [True, False])
def test_forecaster_predicts_horizon_of_targets(
    objective: ForecastObjective, is_instance_normalized: bool, fast_training: TrainingSettings, sine_series: np.ndarray
) -> None:
    forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(16,)),
        window=ForecastWindow(input_length=24, horizon=6, stride=3),
        objective=objective,
        training=fast_training,
        target_channels=(1, 0),
        is_instance_normalized=is_instance_normalized,
    ).build()
    history = forecaster.fit(sine_series[:300], sine_series[300:])
    assert history.num_epochs == 3
    assert len(history.validation_losses) == 3
    prediction = forecaster.forecast(sine_series)
    assert prediction.point.shape == (1, 6, 2)
    assert prediction.quantiles.shape == (1, 6, 2, len(objective.quantile_levels))
    metrics = forecaster.evaluate(sine_series[300:])
    assert metrics.num_windows == (100 - 30) // 3 + 1
    assert (metrics.quantile_loss is None) == (not objective.quantile_levels)


def test_forecasts_are_in_original_scale(sine_series: np.ndarray) -> None:
    shifted = sine_series * 100.0 + 1000.0
    forecaster = ForecasterSettings(
        model=DLinearParameters(kernel_size=5),
        window=ForecastWindow(input_length=32, horizon=8),
        training=TrainingSettings(max_epochs=20, early_stopping_patience=None, device=Device.CPU),
        target_channels=(0,),
    ).build()
    _ = forecaster.fit(shifted[:300])
    metrics = forecaster.evaluate(shifted[300:])
    assert metrics.mae < 20.0
    assert np.all(np.abs(forecaster.forecast(shifted).point - 1000.0) < 200.0)


def test_quantile_forecasts_are_ordered(fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    forecaster = ForecasterSettings(
        model=RNNParameters(hidden_size=8, num_layers=1),
        window=ForecastWindow(input_length=16, horizon=4, stride=4),
        objective=QuantileObjective((0.05, 0.5, 0.95)),
        training=fast_training,
    ).build()
    _ = forecaster.fit(sine_series)
    prediction = forecaster.predict(np.stack([sine_series[:16], sine_series[100:116]]))
    assert np.all(np.diff(prediction.quantiles, axis=-1) >= 0.0)
    assert np.array_equal(prediction.get_quantile(0.5), prediction.point)
    with pytest.raises(KeyError):
        _ = prediction.get_quantile(0.3)


def test_forecaster_rejects_invalid_input(fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    settings = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        window=ForecastWindow(input_length=16, horizon=4),
        training=fast_training,
    )
    forecaster = settings.build()
    with pytest.raises(NotFittedError):
        _ = forecaster.forecast(sine_series)
    with pytest.raises(ValueError, match="at least 20 steps"):
        _ = forecaster.fit(sine_series[:10])
    with pytest.raises(ValueError, match="NaN"):
        _ = forecaster.fit(np.full((50, 3), np.nan))
    _ = forecaster.fit(sine_series)
    with pytest.raises(ValueError, match="channels"):
        _ = forecaster.predict(np.zeros((2, 16, 2)))
    with pytest.raises(ValueError, match="target_channels"):
        _ = (
            ForecasterSettings(
                model=MLPParameters(), window=ForecastWindow(8, 2), target_channels=(5,), training=fast_training
            )
            .build()
            .fit(sine_series)
        )
    with pytest.raises(ValueError, match="repeat"):
        _ = ForecasterSettings(model=MLPParameters(), window=ForecastWindow(8, 2), target_channels=(0, 0))


def test_forecaster_trains_on_several_series(fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    forecaster = ForecasterSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        window=ForecastWindow(input_length=16, horizon=4),
        training=fast_training,
    ).build()
    _ = forecaster.fit([sine_series[:150], sine_series[150:]])
    assert forecaster.evaluate([sine_series[:50], sine_series[50:90]]).num_windows == (50 - 19) + (40 - 19)
