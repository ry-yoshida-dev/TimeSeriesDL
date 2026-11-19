import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import DLinearParameters
from time_series_dl.tasks.forecasting import (
    ConformalSettings,
    ConformityScore,
    Forecaster,
    ForecasterSettings,
    ForecastWindow,
    QuantileObjective,
)
from time_series_dl.tasks.forecasting.objective import ForecastObjective

HORIZON: int = 3


def _noisy_sine(length: int, noise: float, seed: int) -> np.ndarray:
    steps = np.arange(length)
    rng = np.random.default_rng(seed)
    return (np.sin(steps / 6.0) + noise * rng.normal(size=length))[:, np.newaxis]


def _fit(objective: ForecastObjective) -> Forecaster:
    forecaster = ForecasterSettings(
        model=DLinearParameters(kernel_size=5),
        window=ForecastWindow(input_length=24, horizon=HORIZON),
        objective=objective,
        training=TrainingSettings(max_epochs=5, batch_size=64, device=Device.CPU, early_stopping_patience=None),
    ).build()
    _ = forecaster.fit(_noisy_sine(800, 0.3, 0))
    return forecaster


@pytest.mark.parametrize(
    ("score", "objective"),
    [
        (ConformityScore.ABSOLUTE_RESIDUAL, QuantileObjective((0.5,))),
        (ConformityScore.QUANTILE_RESIDUAL, QuantileObjective((0.05, 0.1, 0.5, 0.9, 0.95))),
    ],
)
def test_split_conformal_coverage_is_close_to_nominal(score: ConformityScore, objective: ForecastObjective) -> None:
    conformal = ConformalSettings(coverage_levels=(0.8, 0.9), score=score).build(_fit(objective))
    conformal.calibrate(_noisy_sine(1000, 0.3, 1))
    assert conformal.num_calibration_windows == 1000 - 24 - HORIZON + 1
    metrics = conformal.evaluate(_noisy_sine(2000, 0.3, 2))
    assert metrics.interval_coverage is not None
    assert metrics.interval_coverage[0.9] == pytest.approx(0.9, abs=0.03)
    assert metrics.interval_coverage[0.8] == pytest.approx(0.8, abs=0.03)
    prediction = conformal.forecast(_noisy_sine(100, 0.3, 3))
    assert prediction.quantile_levels == pytest.approx((0.05, 0.1, 0.9, 0.95))
    assert np.all(np.diff(prediction.quantiles, axis=-1) >= 0.0)


def test_adaptive_conformal_inference_recovers_coverage_under_shift() -> None:
    conformal = ConformalSettings(coverage_levels=(0.9,), adaptation_rate=0.05).build(_fit(QuantileObjective((0.5,))))
    conformal.calibrate(_noisy_sine(1000, 0.3, 1))
    shifted = _noisy_sine(3000, 0.6, 2)
    static = conformal.evaluate(shifted)
    online = conformal.evaluate(shifted, is_online=True)
    assert static.interval_coverage is not None and online.interval_coverage is not None
    assert static.interval_coverage[0.9] < 0.8
    assert online.interval_coverage[0.9] == pytest.approx(0.9, abs=0.03)
    assert float(conformal.miscoverage_rates.mean()) < 0.1
    # Calibrating again resets the rates; update adapts them like the online evaluation.
    conformal.calibrate(_noisy_sine(1000, 0.3, 1))
    assert np.allclose(conformal.miscoverage_rates, 0.1)
    past = np.stack([shifted[start : start + 24] for start in range(0, 600, 3)])
    future = np.stack([shifted[start + 24 : start + 24 + HORIZON] for start in range(0, 600, 3)])
    used = conformal.update(past, future)
    assert used.quantiles.shape == (200, HORIZON, 1, 2)
    assert float(conformal.miscoverage_rates.mean()) < 0.1


def test_conformal_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        _ = ConformalSettings(coverage_levels=(0.9, 0.8))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        _ = ConformalSettings(coverage_levels=(1.0,))
    with pytest.raises(ValueError, match="adaptation_rate"):
        _ = ConformalSettings(adaptation_rate=-0.1)
    forecaster = _fit(QuantileObjective((0.1, 0.5, 0.9)))
    conformal = ConformalSettings(coverage_levels=(0.9,), score=ConformityScore.QUANTILE_RESIDUAL).build(forecaster)
    with pytest.raises(NotFittedError):
        _ = conformal.forecast(_noisy_sine(50, 0.3, 0))
    with pytest.raises(ValueError, match="quantile residual"):
        conformal.calibrate(_noisy_sine(200, 0.3, 1))
    small = ConformalSettings(coverage_levels=(0.99,)).build(forecaster)
    small.calibrate(_noisy_sine(40, 0.3, 1))
    # 14 calibration windows cannot certify 99% coverage, so the interval is unbounded.
    assert np.all(np.isinf(small.forecast(_noisy_sine(50, 0.3, 0)).quantiles))
