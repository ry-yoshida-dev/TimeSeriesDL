import numpy as np
import pytest

from time_series_dl import Device, TrainingSettings


@pytest.fixture
def fast_training() -> TrainingSettings:
    return TrainingSettings(max_epochs=3, batch_size=16, device=Device.CPU, early_stopping_patience=None)


@pytest.fixture
def sine_series() -> np.ndarray:
    steps = np.arange(400)
    rng = np.random.default_rng(0)
    return np.stack(
        [np.sin(steps / 8.0), np.cos(steps / 13.0) * 2.0 + 3.0, rng.normal(size=steps.size)], axis=1
    ).astype(np.float64)
