import numpy as np
import pytest

from time_series_dl import Device, HuberLoss, TrainingSettings
from time_series_dl.models import RNNParameters
from time_series_dl.tasks.regression import RegressorSettings


def make_amplitudes(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(1.0, 10.0, size)
    offset = rng.uniform(-5.0, 5.0, size)
    steps = np.arange(24)
    values = amplitude[:, None] * np.sin(steps[None, :] / 3.0) + offset[:, None]
    return values[:, :, None], np.stack([amplitude, offset], axis=1)


def test_regressor_learns_several_outputs() -> None:
    X, y = make_amplitudes(300, seed=0)
    X_test, y_test = make_amplitudes(60, seed=1)
    regressor = RegressorSettings(
        model=RNNParameters(hidden_size=32, num_layers=1),
        training=TrainingSettings(max_epochs=40, device=Device.CPU, learning_rate=5e-3),
        loss=HuberLoss(delta=0.5),
    ).build()
    _ = regressor.fit(X, y, validation=(X_test, y_test))
    predictions = regressor.predict(X_test)
    assert predictions.shape == (60, 2)
    metrics = regressor.evaluate(X_test, y_test)
    assert len(metrics) == 2
    assert all(values.r2 is not None and values.r2 > 0.9 for values in metrics)


def test_regressor_keeps_one_dimensional_targets(fast_training: TrainingSettings) -> None:
    X, y = make_amplitudes(40, seed=2)
    regressor = RegressorSettings(model=RNNParameters(hidden_size=4, num_layers=1), training=fast_training).build()
    _ = regressor.fit(X, y[:, 0])
    assert regressor.predict(X).shape == (40,)
    assert len(regressor.evaluate(X, y[:, 0])) == 1
    with pytest.raises(ValueError, match="outputs"):
        _ = regressor.evaluate(X, y)
    with pytest.raises(ValueError, match="NaN"):
        _ = regressor.fit(X, np.full(40, np.nan))
