"""Every task loads a saved model whose parameters name published weights without requesting them again."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from data_scaler import NoScalingParameters
from irregular_datasets import make_sinusoids, make_timing_records
from published_mtan import PublishedMTANParameters
from published_ts2vec import PublishedTS2VecParameters
from test_model_moment import TINY, official_state_dict

from time_series_dl import Device, TrainingSettings
from time_series_dl.data.irregular import IrregularSequenceCollection, QueryTimes
from time_series_dl.models.architectures.continuous_time import MTANParameters
from time_series_dl.models.architectures.foundation.moment import MOMENTParameters
from time_series_dl.models.components.pretrained import HubClient, PretrainedWeights, SafetensorsFile
from time_series_dl.persistence import SettingsCodec, restorable_model
from time_series_dl.tasks.anomaly_detection import AnomalyDetector, AnomalyDetectorSettings, ReconstructionErrorScoring
from time_series_dl.tasks.classification import ClassifierSettings, SequenceClassifier
from time_series_dl.tasks.imputation import Imputer, ImputerSettings
from time_series_dl.tasks.irregular.classification import IrregularClassifierSettings, IrregularSequenceClassifier
from time_series_dl.tasks.irregular.interpolation import IrregularInterpolator, IrregularInterpolatorSettings
from time_series_dl.tasks.irregular.regression import IrregularRegressorSettings, IrregularSequenceRegressor
from time_series_dl.tasks.regression import RegressorSettings, SequenceRegressor
from time_series_dl.tasks.representation_learning import RepresentationLearner, RepresentationLearnerSettings

WEIGHTS: PretrainedWeights = PretrainedWeights(repository="AutonLab/MOMENT-1-small", revision="main")
PRETRAINED_MOMENT: MOMENTParameters = replace(TINY, weights=WEIGHTS)
MTAN: PublishedMTANParameters = PublishedMTANParameters(
    hidden_size=8, time_embedding_size=4, num_reference_points=8, latent_size=2, projection_size=8, weights=WEIGHTS
)
TS2VEC: PublishedTS2VecParameters = PublishedTS2VecParameters(
    hidden_size=8, output_size=8, num_blocks=2, weights=WEIGHTS
)
TRAINING: TrainingSettings = TrainingSettings(
    max_epochs=1, batch_size=8, device=Device.CPU, random_seed=0, early_stopping_patience=None
)

type Offline = Callable[[], None]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Offline:
    """
    Serve MOMENT weights under the published names in place of the Hugging Face Hub; the returned function makes
    every later request fail, as without network access or cache.
    """
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, official_state_dict(TINY))

    def download(weights: PretrainedWeights) -> Path:
        assert weights == WEIGHTS
        return path

    def refuse(weights: PretrainedWeights) -> Path:
        raise ConnectionError(f"{weights.repository} was requested while loading a saved task")

    monkeypatch.setattr(HubClient, "download", staticmethod(download))
    return lambda: monkeypatch.setattr(HubClient, "download", staticmethod(refuse))


def register(monkeypatch: pytest.MonkeyPatch, module: str, *roots: type[object]) -> None:
    """Let the codec of a task module read a test-only model class."""
    monkeypatch.setattr(f"{module}._CODEC", SettingsCodec.including(*roots))


def make_sequences(size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    steps = np.arange(24)
    phase = rng.uniform(0.0, 6.0, size)[:, None]
    return np.stack([np.sin(steps / 4.0 + phase), np.cos(steps / 3.0 + phase)], axis=2)


def test_restorable_model_drops_only_published_weights() -> None:
    assert restorable_model(PRETRAINED_MOMENT) == TINY
    plain: MTANParameters = MTANParameters()
    assert restorable_model(plain) is plain
    assert restorable_model(MTAN) == replace(MTAN, weights=None)


def test_imputer(offline: Offline, tmp_path: Path) -> None:
    sequences = make_sequences(16)
    sequences[np.random.default_rng(1).random(sequences.shape) < 0.2] = np.nan
    imputer = ImputerSettings(model=PRETRAINED_MOMENT, scaling=NoScalingParameters(), training=TRAINING).build()
    _ = imputer.fit(sequences)
    path: Path = tmp_path / "imputer.pt"
    imputer.save(path)
    offline()
    np.testing.assert_allclose(Imputer.load(path).impute(sequences), imputer.impute(sequences), atol=1e-5)


def test_anomaly_detector(offline: Offline, sine_series: np.ndarray, tmp_path: Path) -> None:
    detector = AnomalyDetectorSettings(
        model=PRETRAINED_MOMENT,
        window_length=24,
        training_stride=8,
        scoring=ReconstructionErrorScoring(),
        training=TRAINING,
    ).build()
    _ = detector.fit(sine_series)
    path: Path = tmp_path / "detector.pt"
    detector.save(path)
    offline()
    np.testing.assert_allclose(AnomalyDetector.load(path).score(sine_series), detector.score(sine_series), atol=1e-5)


def test_classifier(offline: Offline, tmp_path: Path) -> None:
    sequences = make_sequences(16)
    labels = np.arange(16) % 2
    classifier = ClassifierSettings(model=PRETRAINED_MOMENT, training=TRAINING).build()
    _ = classifier.fit(sequences, labels)
    path: Path = tmp_path / "classifier.pt"
    classifier.save(path)
    offline()
    restored = SequenceClassifier.load(path)
    np.testing.assert_allclose(
        restored.predict_proba(sequences).values, classifier.predict_proba(sequences).values, atol=1e-5
    )


def test_regressor(offline: Offline, tmp_path: Path) -> None:
    sequences = make_sequences(16)
    targets = sequences[:, -1, 0]
    regressor = RegressorSettings(model=PRETRAINED_MOMENT, training=TRAINING).build()
    _ = regressor.fit(sequences, targets)
    path: Path = tmp_path / "regressor.pt"
    regressor.save(path)
    offline()
    np.testing.assert_allclose(SequenceRegressor.load(path).predict(sequences), regressor.predict(sequences), atol=1e-5)


def test_representation_learner(offline: Offline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    register(
        monkeypatch,
        "time_series_dl.tasks.representation_learning.learner",
        RepresentationLearnerSettings,
        PublishedTS2VecParameters,
    )
    sequences = make_sequences(16)
    learner = RepresentationLearnerSettings(method=TS2VEC, training=TRAINING).build()
    _ = learner.fit(sequences)
    path: Path = tmp_path / "learner.pt"
    learner.save(path)
    offline()
    restored = RepresentationLearner.load(path)
    np.testing.assert_allclose(restored.encode(sequences), learner.encode(sequences), atol=1e-5)
    _ = restored.build_transfer_network(24, 2)


def test_irregular_classifier(offline: Offline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    register(
        monkeypatch,
        "time_series_dl.tasks.irregular.classification.classifier",
        IrregularClassifierSettings,
        PublishedMTANParameters,
    )
    X, y = make_timing_records(16, seed=0)
    classifier = IrregularClassifierSettings(model=MTAN, training=TRAINING).build()
    _ = classifier.fit(X, y)
    path: Path = tmp_path / "classifier.pt"
    classifier.save(path)
    offline()
    restored = IrregularSequenceClassifier.load(path)
    np.testing.assert_allclose(restored.predict_proba(X).values, classifier.predict_proba(X).values, atol=1e-5)


def test_irregular_regressor(offline: Offline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    register(
        monkeypatch,
        "time_series_dl.tasks.irregular.regression.regressor",
        IrregularRegressorSettings,
        PublishedMTANParameters,
    )
    X, y = make_timing_records(16, seed=0)
    regressor = IrregularRegressorSettings(model=MTAN, training=TRAINING).build()
    _ = regressor.fit(X, y.astype(np.float64))
    path: Path = tmp_path / "regressor.pt"
    regressor.save(path)
    offline()
    np.testing.assert_allclose(IrregularSequenceRegressor.load(path).predict(X), regressor.predict(X), atol=1e-5)


def test_irregular_interpolator(offline: Offline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    register(
        monkeypatch,
        "time_series_dl.tasks.irregular.interpolation.interpolator",
        IrregularInterpolatorSettings,
        PublishedMTANParameters,
    )
    X: IrregularSequenceCollection = make_sinusoids(16, seed=0)
    interpolator = IrregularInterpolatorSettings(model=MTAN, training=TRAINING).build()
    _ = interpolator.fit(X)
    path: Path = tmp_path / "interpolator.pt"
    interpolator.save(path)
    offline()
    restored = IrregularInterpolator.load(path)
    query_times = QueryTimes.from_arrays([np.linspace(0.0, 8.0, 3)] * 16)
    for original, loaded in zip(interpolator.predict(X, query_times), restored.predict(X, query_times), strict=True):
        np.testing.assert_allclose(loaded, original, atol=1e-5)
