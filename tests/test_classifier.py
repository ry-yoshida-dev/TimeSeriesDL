from typing import cast

import numpy as np
import pytest
from deep_learning_loss import CrossEntropyLoss, FocalLoss
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.array_types import LabelArray
from time_series_dl.models import MLPParameters, TCNParameters
from time_series_dl.tasks.classification import ClassifierSettings


def make_classes(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.array(["low", "high", "mid"]), size=size)
    frequency = np.select([labels == "low", labels == "mid"], [0.2, 0.6], 1.2)
    steps = np.arange(32)
    values = np.sin(frequency[:, None] * steps[None, :] + rng.uniform(0, 6, size)[:, None])
    return np.stack([values, rng.normal(scale=0.1, size=values.shape)], axis=2), labels


def test_classifier_learns_frequencies() -> None:
    X, y = make_classes(240, seed=0)
    X_test, y_test = make_classes(90, seed=1)
    classifier = ClassifierSettings(
        model=TCNParameters(channel_sizes=(16, 16, 16)),
        training=TrainingSettings(max_epochs=30, device=Device.CPU, learning_rate=3e-3),
    ).build()
    history = classifier.fit(X, y, validation=(X_test, y_test))
    assert history.num_epochs >= 1
    assert classifier.classes.tolist() == ["high", "low", "mid"]
    scores = classifier.predict_proba(X_test)
    assert scores.values.shape == (90, 3)
    assert np.allclose(scores.values.sum(axis=1), 1.0)
    metrics = classifier.evaluate(X_test, y_test)
    assert metrics.accuracy > 0.9
    assert metrics.log_loss is not None


def test_classifier_balances_classes_and_validates(fast_training: TrainingSettings) -> None:
    X, y = make_classes(40, seed=2)
    integer_labels = (y == "low").astype(np.int64)
    classifier = ClassifierSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        training=fast_training,
        is_class_balanced=True,
        loss=CrossEntropyLoss(label_smoothing=0.1),
    ).build()
    with pytest.raises(NotFittedError):
        _ = classifier.predict(X)
    _ = classifier.fit(X, integer_labels)
    assert set(classifier.predict(X).tolist()) <= {0, 1}
    with pytest.raises(ValueError, match="not fitted on"):
        _ = classifier.evaluate(X, np.full(40, 7))
    with pytest.raises(ValueError, match="at least 2 classes"):
        _ = classifier.fit(X, np.zeros(40, dtype=np.int64))
    with pytest.raises(TypeError, match="labels"):
        _ = classifier.fit(X, cast(LabelArray, np.zeros(40)))
    with pytest.raises(ValueError, match="length 32"):
        _ = classifier.predict(X[:, :10])


def test_classifier_trains_with_focal_loss(fast_training: TrainingSettings) -> None:
    X, y = make_classes(40, seed=3)
    classifier = ClassifierSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        training=fast_training,
        is_class_balanced=True,
        loss=FocalLoss(gamma=2.0),
    ).build()
    history = classifier.fit(X, y)
    assert all(np.isfinite(history.training_losses))
    assert classifier.predict_proba(X).values.shape == (40, 3)
