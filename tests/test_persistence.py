import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from data_scaler import MinMaxScalingParameters, RobustScalingParameters
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, HuberLoss, TrainingSettings
from time_series_dl.models import (
    MLPParameters,
    ModelArchitecture,
    NBEATSBlockKind,
    NBEATSParameters,
    NBEATSStack,
    RNNParameters,
    TCNParameters,
)
from time_series_dl.persistence import Checkpoint, EncodedValue, SettingsCodec, TaskKind
from time_series_dl.tasks.anomaly_detection import AnomalyDetector, AnomalyDetectorSettings
from time_series_dl.tasks.classification import ClassifierSettings, FocalLoss, SequenceClassifier
from time_series_dl.tasks.forecasting import (
    Forecaster,
    ForecasterSettings,
    ForecastWindow,
    PointObjective,
    QuantileObjective,
)
from time_series_dl.tasks.imputation import Imputer, ImputerSettings
from time_series_dl.tasks.regression import RegressorSettings, SequenceRegressor


def test_forecaster_round_trip(tmp_path: Path, fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    forecaster = ForecasterSettings(
        model=TCNParameters(channel_sizes=(8, 8)),
        window=ForecastWindow(input_length=24, horizon=6, stride=2),
        objective=QuantileObjective((0.1, 0.5, 0.9)),
        target_channels=(2, 0),
        scaling=RobustScalingParameters(quantile_range=(10.0, 90.0)),
        training=fast_training,
    ).build()
    with pytest.raises(NotFittedError):
        forecaster.save(tmp_path / "unfitted.pt")
    _ = forecaster.fit(sine_series)
    path = tmp_path / "forecaster.pt"
    forecaster.save(path)
    loaded = Forecaster.load(path)
    assert loaded.settings == forecaster.settings
    assert loaded.history == forecaster.history
    assert loaded.target_channels == (2, 0)
    expected = forecaster.forecast(sine_series)
    actual = loaded.forecast(sine_series)
    np.testing.assert_allclose(actual.point, expected.point)
    np.testing.assert_allclose(actual.quantiles, expected.quantiles)
    assert loaded.evaluate(sine_series).mae == pytest.approx(forecaster.evaluate(sine_series).mae)


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_forecaster_round_trip_of_every_model(
    tmp_path: Path, fast_training: TrainingSettings, sine_series: np.ndarray, architecture: ModelArchitecture
) -> None:
    forecaster = ForecasterSettings(
        model=architecture.parameters_type(),
        window=ForecastWindow(input_length=48, horizon=8, stride=8),
        objective=PointObjective(loss=HuberLoss(delta=0.5)),
        training=replace(fast_training, max_epochs=1),
    ).build()
    _ = forecaster.fit(sine_series)
    path = tmp_path / f"{architecture.value}.pt"
    forecaster.save(path)
    loaded = Forecaster.load(path)
    # Informer samples keys at random in every forward pass.
    _ = torch.manual_seed(0)
    expected = forecaster.forecast(sine_series).point
    _ = torch.manual_seed(0)
    np.testing.assert_allclose(loaded.forecast(sine_series).point, expected, rtol=1e-5, atol=1e-6)


def test_classifier_round_trip_keeps_string_classes(tmp_path: Path, fast_training: TrainingSettings) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 16, 2))
    y = rng.choice(np.array(["cat", "dog", "eel"]), size=30)
    classifier = ClassifierSettings(
        model=MLPParameters(hidden_sizes=(8,)),
        training=fast_training,
        loss=FocalLoss(gamma=1.5),
        is_class_balanced=True,
    ).build()
    _ = classifier.fit(X, y)
    path = tmp_path / "classifier.pt"
    classifier.save(path)
    loaded = SequenceClassifier.load(path)
    assert loaded.settings == classifier.settings
    assert loaded.classes.tolist() == ["cat", "dog", "eel"]
    assert loaded.classes.dtype == classifier.classes.dtype
    assert not loaded.classes.flags.writeable
    np.testing.assert_allclose(loaded.predict_proba(X).values, classifier.predict_proba(X).values)
    assert loaded.predict(X).tolist() == classifier.predict(X).tolist()


def test_classifier_round_trip_keeps_integer_classes(tmp_path: Path, fast_training: TrainingSettings) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 8, 1))
    y = rng.choice(np.array([3, 7], dtype=np.int32), size=20)
    classifier = ClassifierSettings(model=MLPParameters(hidden_sizes=(4,)), training=fast_training).build()
    _ = classifier.fit(X, y)
    path = tmp_path / "classifier.pt"
    classifier.save(path)
    loaded = SequenceClassifier.load(path)
    assert loaded.classes.dtype == np.int32
    assert loaded.predict(X).tolist() == classifier.predict(X).tolist()


def test_regressor_round_trip(tmp_path: Path, fast_training: TrainingSettings) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 12, 2))
    y = np.stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2))], axis=1)
    regressor = RegressorSettings(
        model=RNNParameters(hidden_size=8),
        training=fast_training,
        loss=HuberLoss(delta=0.5),
        target_scaling=MinMaxScalingParameters(feature_range=(-1.0, 1.0)),
    ).build()
    _ = regressor.fit(X, y)
    path = tmp_path / "regressor.pt"
    regressor.save(path)
    loaded = SequenceRegressor.load(path)
    assert loaded.settings == regressor.settings
    np.testing.assert_allclose(loaded.predict(X), regressor.predict(X))
    assert loaded.predict(X).shape == (30, 2)


def test_imputer_round_trip(tmp_path: Path, fast_training: TrainingSettings) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 16, 2))
    X[rng.random(X.shape) < 0.1] = np.nan
    imputer = ImputerSettings(model=MLPParameters(hidden_sizes=(8,)), training=fast_training, mask_ratio=0.3).build()
    _ = imputer.fit(X)
    path = tmp_path / "imputer.pt"
    imputer.save(path)
    loaded = Imputer.load(path)
    assert loaded.settings == imputer.settings
    np.testing.assert_allclose(loaded.impute(X), imputer.impute(X))


def test_anomaly_detector_round_trip(tmp_path: Path, fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(8,)), window_length=16, training_stride=4, training=fast_training
    ).build()
    _ = detector.fit(sine_series)
    path = tmp_path / "detector.pt"
    detector.save(path)
    loaded = AnomalyDetector.load(path)
    assert loaded.threshold == detector.threshold
    np.testing.assert_allclose(loaded.score(sine_series), detector.score(sine_series))
    assert loaded.detect(sine_series).tolist() == detector.detect(sine_series).tolist()


def test_load_overrides_device(tmp_path: Path, fast_training: TrainingSettings, sine_series: np.ndarray) -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(4,)), window_length=16, training_stride=8, training=fast_training
    ).build()
    _ = detector.fit(sine_series)
    path = tmp_path / "detector.pt"
    detector.save(path)
    checkpoint = Checkpoint.load(path, TaskKind.ANOMALY_DETECTION)
    cuda_training = replace(fast_training, device=Device.CUDA)
    cuda_settings = SettingsCodec.including(AnomalyDetectorSettings).encode(
        replace(detector.settings, training=cuda_training)
    )
    replace(checkpoint, settings=cuda_settings).save(path)
    loaded = AnomalyDetector.load(path, device=Device.CPU)
    assert loaded.settings.training.device == Device.CPU
    np.testing.assert_allclose(loaded.score(sine_series), detector.score(sine_series))


def test_load_rejects_checkpoint_of_another_task(
    tmp_path: Path, fast_training: TrainingSettings, sine_series: np.ndarray
) -> None:
    detector = AnomalyDetectorSettings(
        model=MLPParameters(hidden_sizes=(4,)), window_length=16, training_stride=8, training=fast_training
    ).build()
    _ = detector.fit(sine_series)
    path = tmp_path / "detector.pt"
    detector.save(path)
    with pytest.raises(ValueError, match="'anomaly_detection' task, not 'imputation'"):
        _ = Imputer.load(path)


def test_load_rejects_pickled_objects(tmp_path: Path) -> None:
    path = tmp_path / "pickled.pt"
    torch.save({"format_version": 1, "task": "imputation", "settings": ForecastWindow(4, 2)}, path)
    with pytest.raises(pickle.UnpicklingError):
        _ = Imputer.load(path)


def test_load_rejects_unsupported_format_version(tmp_path: Path) -> None:
    path = tmp_path / "future.pt"
    torch.save({"format_version": 999, "task": "imputation"}, path)
    with pytest.raises(ValueError, match="format version 999"):
        _ = Imputer.load(path)


def test_settings_codec_keeps_nested_implementations() -> None:
    codec = SettingsCodec.including(ForecasterSettings, PointObjective)
    settings = ForecasterSettings(
        model=NBEATSParameters(
            stacks=(NBEATSStack(kind=NBEATSBlockKind.TREND, num_blocks=2), NBEATSStack(kind=NBEATSBlockKind.GENERIC))
        ),
        window=ForecastWindow(input_length=12, horizon=3),
        objective=PointObjective(loss=HuberLoss(delta=0.3)),
        target_channels=(1,),
        training=TrainingSettings(early_stopping_patience=None, device=Device.CPU),
    )
    encoded = codec.encode(settings)
    assert codec.decode(ForecasterSettings, encoded) == settings


def test_settings_codec_rejects_unknown_classes() -> None:
    @dataclass(frozen=True)
    class Unregistered:
        value: int = 0

    codec = SettingsCodec.including(ForecastWindow)
    with pytest.raises(TypeError, match="not a known settings class"):
        _ = codec.encode(Unregistered())
    forged: EncodedValue = {"type": "Unregistered", "fields": {"value": 1}}
    with pytest.raises(ValueError, match="unknown settings class 'Unregistered'"):
        _ = codec.decode(ForecastWindow, forged)
    window: EncodedValue = codec.encode(ForecastWindow(4, 2))
    with pytest.raises(TypeError, match="expected TrainingSettings"):
        _ = codec.decode(TrainingSettings, window)
