from pathlib import Path

import numpy as np
from irregular_datasets import make_timing_records

from time_series_dl import Device, HuberLoss, TrainingSettings
from time_series_dl.models.architectures.continuous_time import LatentODEParameters, ODERNNParameters
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.tasks.irregular.regression import IrregularRegressorSettings, IrregularSequenceRegressor


def test_regressor_learns_the_time_of_the_step() -> None:
    X, _ = make_timing_records(200, seed=0)
    X_test, _ = make_timing_records(100, seed=1)
    targets: np.ndarray = np.array([sequence.times[sequence.values[:, 0] > 0.5][0] for sequence in X.sequences])
    test_targets: np.ndarray = np.array(
        [sequence.times[sequence.values[:, 0] > 0.5][0] for sequence in X_test.sequences]
    )
    regressor = IrregularRegressorSettings(
        model=ODERNNParameters(hidden_size=16, ode_hidden_sizes=(16,), head=IrregularEncoderHead(hidden_sizes=(16,))),
        training=TrainingSettings(max_epochs=30, learning_rate=1e-2, device=Device.CPU, early_stopping_patience=None),
    ).build()
    _ = regressor.fit(X, targets)
    predictions: np.ndarray = regressor.predict(X_test)
    assert predictions.shape == (100,)
    metrics = regressor.evaluate(X_test, test_targets)
    assert len(metrics) == 1
    r2: float | None = metrics[0].r2
    assert r2 is not None and r2 > 0.6


def test_regressor_round_trips_with_vector_targets(tmp_path: Path, fast_training: TrainingSettings) -> None:
    X, labels = make_timing_records(30, seed=3)
    targets: np.ndarray = np.stack([labels.astype(np.float64), X.lengths.astype(np.float64)], axis=1)
    regressor = IrregularRegressorSettings(
        model=LatentODEParameters(
            latent_size=2, recognition_size=4, recognition_hidden_sizes=(4,), dynamics_hidden_sizes=(4,)
        ),
        training=fast_training,
        loss=HuberLoss(delta=0.5),
    ).build()
    _ = regressor.fit(X, targets, validation=(X, targets))
    predictions: np.ndarray = regressor.predict(X)
    assert predictions.shape == (30, 2)
    path: Path = tmp_path / "regressor.pt"
    regressor.save(path)
    restored = IrregularSequenceRegressor.load(path)
    assert np.allclose(restored.predict(X), predictions, atol=1e-5)
    assert len(restored.evaluate(X, targets)) == 2
