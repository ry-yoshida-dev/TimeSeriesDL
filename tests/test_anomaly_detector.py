from pathlib import Path

import numpy as np
import pytest
import torch
from data_scaler import MinMaxScalingParameters, NoScalingParameters, ScalingParameters, StandardScalingParameters
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from time_series_dl import Device, MAELoss, TrainingSettings
from time_series_dl.models import (
    DLinearParameters,
    ForecastingModel,
    MLPParameters,
    NBEATSParameters,
    ReconstructionModel,
)
from time_series_dl.models.architectures.mlp.usad import USADParameters
from time_series_dl.models.architectures.transformer.anomaly_transformer import AnomalyTransformerParameters
from time_series_dl.models.architectures.transformer.dcdetector import DCdetectorParameters
from time_series_dl.models.architectures.transformer.tranad import TranADParameters
from time_series_dl.models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from time_series_dl.tasks.anomaly_detection import (
    AnomalyDetectionMetrics,
    AnomalyDetector,
    AnomalyDetectorSettings,
    AnomalyScoring,
    ForecastErrorScoring,
    NativeScoring,
    ReconstructionErrorScoring,
)
from time_series_dl.training import EpochScheduledObjective, Trainer


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
    assert metrics.point_adjusted_recall == 1.0
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
    assert metrics.point_adjusted_precision == pytest.approx(0.75)
    assert metrics.point_adjusted_recall == pytest.approx(0.6)
    assert metrics.roc_auc is not None
    assert AnomalyDetectionMetrics.compute(labels & False, detections, scores).roc_auc is None


def test_range_based_metrics() -> None:
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    detections = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    metrics = AnomalyDetectionMetrics.compute(labels, detections, np.zeros(9))
    # Labeled ranges are 1/3 and 0 covered; detected ranges are fully and not anomalous.
    assert metrics.range_recall == pytest.approx(1.0 / 6.0)
    assert metrics.range_precision == pytest.approx(0.5)
    assert metrics.range_f1 == pytest.approx(0.25)
    # One labeled range overlapped by two detected ranges is penalized by the cardinality factor.
    split = AnomalyDetectionMetrics.compute(
        np.array([1, 1, 1, 1, 0], dtype=bool), np.array([1, 0, 1, 0, 0], dtype=bool), np.zeros(5)
    )
    assert split.range_recall == pytest.approx(0.25)
    assert split.range_precision == pytest.approx(1.0)
    assert split.point_adjusted_recall == pytest.approx(1.0)
    nothing = AnomalyDetectionMetrics.compute(labels, np.zeros(9, dtype=bool), np.zeros(9))
    assert (nothing.range_precision, nothing.range_recall, nothing.range_f1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (MLPParameters(), ReconstructionErrorScoring),
        (NBEATSParameters(), ForecastErrorScoring),
        (USADParameters(), NativeScoring),
        (DCdetectorParameters(), NativeScoring),
    ],
)
def test_default_scoring_follows_the_model(
    model: ReconstructionModel | ForecastingModel | AnomalyScoringModel, expected: type[AnomalyScoring]
) -> None:
    assert isinstance(AnomalyDetectorSettings(model=model, window_length=15).resolved_scoring, expected)


def test_scoring_must_fit_the_model() -> None:
    with pytest.raises(TypeError, match="does not support ReconstructionErrorScoring"):
        _ = AnomalyDetectorSettings(model=USADParameters(), window_length=8, scoring=ReconstructionErrorScoring())
    with pytest.raises(TypeError, match="does not support NativeScoring"):
        _ = AnomalyDetectorSettings(model=MLPParameters(), window_length=8, scoring=NativeScoring())
    with pytest.raises(ValueError, match="horizon"):
        _ = ForecastErrorScoring(horizon=0)


def test_forecast_error_scoring_finds_injected_spike() -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(32,)),
        window_length=20,
        scoring=ForecastErrorScoring(),
        training=TrainingSettings(max_epochs=30, device=Device.CPU, learning_rate=3e-3),
    ).build()
    _ = detector.fit(make_series(800, seed=0))
    test = make_series(300, seed=2)
    test[150, 1] += 3.0
    scores = detector.score(test)
    assert scores.shape == (300,)
    assert int(np.argmax(scores)) == 150
    assert scores[150] > 10.0 * np.median(scores)
    # The first window has no forecast and gets the lowest score.
    assert np.all(scores[:20] == scores.min())
    assert detector.detect(test)[150]


def test_forecast_error_score_is_the_one_step_error(fast_training: TrainingSettings) -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(4,)),
        window_length=8,
        scoring=ForecastErrorScoring(is_instance_normalized=False),
        scaling=NoScalingParameters(),
        training=fast_training,
    ).build()
    series = make_series(40, seed=0)
    _ = detector.fit(series)
    past = torch.from_numpy(series[np.newaxis, 10:18].astype(np.float32))
    with torch.no_grad():
        forecast: Tensor = detector.network(past)[0, 0, :, 0]
    target: Tensor = torch.from_numpy(series[18].astype(np.float32))
    expected: float = (forecast - target).square().mean().item()
    assert detector.score(series)[18] == pytest.approx(expected, rel=1e-4)


def test_forecast_error_scoring_averages_horizon_errors(fast_training: TrainingSettings) -> None:
    detector = AnomalyDetectorSettings(
        model=DLinearParameters(), window_length=12, scoring=ForecastErrorScoring(horizon=3), training=fast_training
    ).build()
    with pytest.raises(ValueError, match="at least 15 steps"):
        _ = detector.fit(make_series(14, seed=0))
    _ = detector.fit(make_series(60, seed=0))
    scores = detector.score(make_series(15, seed=1))
    assert scores.shape == (15,)
    assert np.all(scores[:12] == scores.min())


ANOMALY_MODELS: list[
    tuple[ReconstructionModel | ForecastingModel | AnomalyScoringModel, AnomalyScoring, ScalingParameters]
] = [
    (MLPParameters(hidden_sizes=(8,)), ReconstructionErrorScoring(), StandardScalingParameters()),
    (DLinearParameters(), ForecastErrorScoring(horizon=2), StandardScalingParameters()),
    (USADParameters(latent_size_per_step=2), NativeScoring(), MinMaxScalingParameters()),
    (TranADParameters(), NativeScoring(), MinMaxScalingParameters()),
    (
        AnomalyTransformerParameters(model_size=8, num_heads=2, num_layers=1, feedforward_size=8),
        NativeScoring(),
        StandardScalingParameters(),
    ),
    (
        DCdetectorParameters(patch_sizes=(2, 5), model_size=8, num_heads=1, num_layers=1),
        NativeScoring(),
        StandardScalingParameters(),
    ),
]


@pytest.mark.parametrize(("model", "scoring", "scaling"), ANOMALY_MODELS)
def test_detector_runs_end_to_end_with_every_kind_of_model(
    model: ReconstructionModel | ForecastingModel | AnomalyScoringModel,
    scoring: AnomalyScoring,
    scaling: ScalingParameters,
    fast_training: TrainingSettings,
    tmp_path: Path,
) -> None:
    detector = AnomalyDetectorSettings(
        model=model, window_length=10, training_stride=3, scoring=scoring, scaling=scaling, training=fast_training
    ).build()
    history = detector.fit(make_series(120, seed=0), validation_series=make_series(40, seed=1))
    assert history.num_epochs == fast_training.max_epochs
    assert all(np.isfinite(history.training_losses)) and all(np.isfinite(history.validation_losses))
    test = make_series(50, seed=2)
    is_anomalous = np.zeros(50, dtype=bool)
    is_anomalous[30:33] = True
    test[30:33] += 3.0
    scores = detector.score(test)
    assert scores.shape == (50,) and bool(np.all(np.isfinite(scores)))
    metrics = detector.evaluate(test, is_anomalous)
    assert 0.0 <= metrics.range_f1 <= 1.0
    path = tmp_path / "detector.pt"
    detector.save(path)
    loaded = AnomalyDetector.load(path)
    assert loaded.settings == detector.settings
    assert loaded.threshold == detector.threshold
    np.testing.assert_allclose(loaded.score(test), scores, rtol=1e-6)


class EpochRecordingObjective:
    def __init__(self, network: nn.Linear) -> None:
        self.network: nn.Linear = network
        self.epochs: list[int] = []

    def start_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        outputs: Tensor = self.network(batch[0])
        return outputs.square().mean()


def test_trainer_announces_each_epoch(fast_training: TrainingSettings) -> None:
    network = nn.Linear(2, 1)
    objective = EpochRecordingObjective(network)
    assert isinstance(objective, EpochScheduledObjective)
    _ = Trainer(fast_training).fit(
        network, objective, TensorDataset(torch.randn(8, 2)), TensorDataset(torch.randn(4, 2))
    )
    assert objective.epochs == [0, 1, 2]
