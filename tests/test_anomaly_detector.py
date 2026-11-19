import numpy as np
import pytest
import torch
from data_scaler import NoScalingParameters

from time_series_dl import Device, MAELoss, TrainingSettings
from time_series_dl.models import MLPParameters
from time_series_dl.tasks.anomaly_detection import AnomalyDetectionMetrics, AnomalyDetectorSettings


def make_series(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = np.arange(length)
    return np.stack([np.sin(steps / 5.0), np.cos(steps / 7.0)], axis=1) + rng.normal(scale=0.05, size=(length, 2))


def test_detector_finds_injected_anomaly() -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(32,)),
        window_length=20,
        training_stride=2,
        training=TrainingSettings(max_epochs=40, device=Device.CPU, learning_rate=3e-3),
    ).build()
    _ = detector.fit(make_series(1000, seed=0), validation_series=make_series(200, seed=1))
    test = make_series(300, seed=2)
    is_anomalous = np.zeros(300, dtype=bool)
    is_anomalous[150:155] = True
    test[150:155, 0] += 3.0
    scores = detector.score(test)
    assert scores.shape == (300,)
    assert scores[150:155].mean() > 10.0 * np.median(scores)
    metrics = detector.evaluate(test, is_anomalous)
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.95
    assert metrics.adjusted_recall == 1.0
    assert detector.detect(test)[150:155].any()


def test_detector_validates(fast_training: TrainingSettings) -> None:
    with pytest.raises(ValueError, match="threshold_quantile"):
        _ = AnomalyDetectorSettings(model=MLPParameters(), window_length=4, threshold_quantile=0.0)
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(4,)), window_length=8, training=fast_training
    ).build()
    with pytest.raises(ValueError, match="at least 8 steps"):
        _ = detector.fit(make_series(5, seed=0))
    _ = detector.fit(make_series(40, seed=0))
    with pytest.raises(ValueError, match="channels"):
        _ = detector.score(np.zeros((40, 3)))


def test_detector_scores_with_its_loss(fast_training: TrainingSettings) -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(4,)),
        window_length=8,
        training=fast_training,
        scaling=NoScalingParameters(),
        loss=MAELoss(),
    ).build()
    series = make_series(40, seed=0)
    _ = detector.fit(series)
    windows = torch.from_numpy(series[:8][np.newaxis].astype(np.float32))
    with torch.no_grad():
        reconstructed = detector.network(windows)
    expected = (reconstructed - windows).abs().mean(dim=2).numpy()[0]
    # The first step lies in one window only, so its score is the loss of that window.
    assert detector.score(series)[0] == pytest.approx(float(expected[0]), rel=1e-5)


def test_anomaly_metrics() -> None:
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    detections = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    scores = np.array([0.1, 0.5, 0.9, 0.4, 0.2, 0.1, 0.3, 0.3, 0.8])
    metrics = AnomalyDetectionMetrics.compute(labels, detections, scores)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.2)
    assert metrics.adjusted_precision == pytest.approx(0.75)
    assert metrics.adjusted_recall == pytest.approx(0.6)
    assert metrics.roc_auc is not None
    assert AnomalyDetectionMetrics.compute(labels & False, detections, scores).roc_auc is None
