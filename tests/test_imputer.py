import numpy as np
import pytest

from time_series_dl import Device, HuberLoss, TrainingSettings
from time_series_dl.models import DLinearParameters, TCNParameters
from time_series_dl.tasks.imputation import ImputerSettings


def make_windows(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = np.arange(48)
    phase = rng.uniform(0.0, 6.0, size)[:, None]
    return np.stack([np.sin(steps / 4.0 + phase), np.cos(steps / 4.0 + phase) * 3.0], axis=2)


def test_imputer_fills_missing_values() -> None:
    windows = make_windows(200, seed=0)
    missing = windows.copy()
    missing[np.random.default_rng(1).random(missing.shape) < 0.2] = np.nan
    imputer = ImputerSettings(
        model=TCNParameters(channel_sizes=(16, 16, 16)),
        training=TrainingSettings(max_epochs=30, device=Device.CPU, learning_rate=3e-3),
    ).build()
    _ = imputer.fit(missing, validation=make_windows(40, seed=2))
    filled = imputer.impute(missing)
    is_missing = np.isnan(missing)
    assert not np.any(np.isnan(filled))
    assert np.array_equal(filled[~is_missing], missing[~is_missing])
    test = make_windows(40, seed=3)
    is_hidden = np.random.default_rng(4).random(test.shape) < 0.3
    metrics = imputer.evaluate(test, is_hidden)
    assert metrics.num_values == int(is_hidden.sum())
    assert metrics.mae < 0.3


def test_imputer_validates(fast_training: TrainingSettings) -> None:
    with pytest.raises(ValueError, match="mask_ratio"):
        _ = ImputerSettings(model=DLinearParameters(), mask_ratio=1.0)
    imputer = ImputerSettings(model=DLinearParameters(kernel_size=5), training=fast_training).build()
    windows = make_windows(20, seed=0)
    _ = imputer.fit(windows)
    with pytest.raises(ValueError, match="at least one observed"):
        _ = imputer.evaluate(windows, np.zeros(windows.shape, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        _ = imputer.evaluate(windows, np.zeros((1, 2, 3), dtype=bool))


def test_imputer_trains_with_point_loss(fast_training: TrainingSettings) -> None:
    imputer = ImputerSettings(
        model=DLinearParameters(kernel_size=5), training=fast_training, loss=HuberLoss(0.5)
    ).build()
    history = imputer.fit(make_windows(20, seed=0))
    assert all(np.isfinite(history.training_losses))
