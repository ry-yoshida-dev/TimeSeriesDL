from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import yaml
from sklearn.exceptions import NotFittedError
from torch import Tensor

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import ModelArchitecture, SummaryShape, TSTParameters
from time_series_dl.models.architectures.convolutional.ts2vec import TS2VecParameters
from time_series_dl.models.architectures.convolutional.ts_tcc import TSTCCParameters
from time_series_dl.models.architectures.pretrained import (
    FrozenSequenceEncoder,
    MaskedReconstructionParameters,
    PretrainedEncoderParameters,
    SimMTMParameters,
)
from time_series_dl.models.components.heads import EncodedSummaryNetwork, EncoderHead
from time_series_dl.models.components.layers import TemporalPooling
from time_series_dl.models.interfaces.protocols.pretraining import PretrainingModel
from time_series_dl.models.settings.parameters_reader import ParametersReader
from time_series_dl.tasks.classification import ClassifierSettings, SequenceClassifier
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow
from time_series_dl.tasks.regression import RegressorSettings
from time_series_dl.tasks.representation_learning import RepresentationLearner, RepresentationLearnerSettings

PRESET_DIRECTORY: Path = Path(__file__).parents[1] / "src" / "time_series_dl" / "config"
TS2VEC: TS2VecParameters = TS2VecParameters(hidden_size=16, output_size=16, num_blocks=3)
TS_TCC: TSTCCParameters = TSTCCParameters(
    channel_sizes=(8, 16), kernel_size=3, context_size=8, context_heads=2, context_layers=1, prediction_steps=2
)
TST: TSTParameters = TSTParameters(model_size=8, num_heads=2, num_layers=1, feedforward_size=16)
METHODS: list[PretrainingModel] = [
    TS2VEC,
    TS_TCC,
    SimMTMParameters(encoder=TST, num_masked=2, projection_size=8),
    MaskedReconstructionParameters(encoder=TST),
]
IS_PRETRAINED_ENCODER_REGISTERED: bool = PretrainedEncoderParameters.__name__ in {
    architecture.parameters_type.__name__ for architecture in ModelArchitecture
}


def make_frequencies(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=size)
    frequency = np.where(labels == 0, 0.25, 0.9)
    steps = np.arange(32)
    values = np.sin(frequency[:, None] * steps[None, :] + rng.uniform(0, 6, size)[:, None])
    noisy = values + rng.normal(scale=0.2, size=values.shape)
    return noisy[:, :, None], labels


def training(max_epochs: int = 2) -> TrainingSettings:
    return TrainingSettings(max_epochs=max_epochs, batch_size=16, device=Device.CPU, early_stopping_patience=None)


def fit_learner(method: PretrainingModel, X: np.ndarray, max_epochs: int = 2) -> RepresentationLearner:
    learner = RepresentationLearnerSettings(method=method, training=training(max_epochs)).build()
    _ = learner.fit(X)
    return learner


def encoder_state(classifier: SequenceClassifier) -> dict[str, Tensor]:
    network = classifier.network
    assert isinstance(network, EncodedSummaryNetwork)
    encoder = network.encoder
    if isinstance(encoder, FrozenSequenceEncoder):
        encoder = encoder.encoder
    return encoder.state_dict()


@pytest.mark.parametrize("method", METHODS, ids=lambda method: type(method).__name__)
def test_every_method_encodes_and_reloads(method: PretrainingModel, tmp_path: Path) -> None:
    X, _ = make_frequencies(24, seed=0)
    learner = fit_learner(method, X)
    assert learner.history.num_epochs == 2
    representations = learner.encode(X)
    assert representations.shape == (24, learner.representation_size)
    timestamps = learner.encode_timestamps(X)
    assert timestamps.shape == (24, learner.encoder.output_length, learner.encoder.output_size)
    path = tmp_path / "learner.pt"
    learner.save(path)
    restored = RepresentationLearner.load(path, device=Device.CPU)
    assert np.allclose(restored.encode(X), representations, atol=1e-5)
    assert restored.settings == learner.settings


def test_pretrained_representations_are_linearly_separable() -> None:
    X, y = make_frequencies(160, seed=0)
    X_test, y_test = make_frequencies(80, seed=1)
    learner = fit_learner(TS2VEC, X, max_epochs=5)
    assert learner.history.training_losses[-1] < learner.history.training_losses[0]
    # A least-squares linear classifier on the representations, with a bias column.
    features: np.ndarray = np.hstack([learner.encode(X), np.ones((160, 1))])
    weights: np.ndarray = np.linalg.lstsq(features, 2.0 * y - 1.0, rcond=None)[0]
    test_features: np.ndarray = np.hstack([learner.encode(X_test), np.ones((80, 1))])
    is_correct: np.ndarray = (test_features @ weights > 0) == (y_test == 1)
    accuracy: float = float(is_correct.mean())
    assert accuracy > 0.9


def test_fit_series_pretrains_on_windows() -> None:
    steps = np.arange(200)
    series = np.stack([np.sin(steps / 5.0), np.cos(steps / 9.0)], axis=1)
    learner = RepresentationLearnerSettings(method=TS2VEC, training=training()).build()
    _ = learner.fit_series([series, series[:120]], window_length=20, stride=5, validation_series=series[50:])
    assert learner.encode(series[None, :20]).shape == (1, 16)
    with pytest.raises(ValueError, match="at least 20 steps"):
        _ = learner.fit_series(series[:10], window_length=20)


def test_learner_validates_inputs() -> None:
    X, _ = make_frequencies(8, seed=0)
    learner = RepresentationLearnerSettings(method=TS2VEC, training=training()).build()
    with pytest.raises(NotFittedError):
        _ = learner.encode(X)
    with pytest.raises(ValueError, match="length 32"):
        _ = learner.fit(X, validation=X[:, :10])
    _ = learner.fit(X)
    with pytest.raises(ValueError, match="length 32"):
        _ = learner.encode(X[:, :10])


@pytest.mark.parametrize("is_encoder_frozen", [True, False])
def test_transfer_to_classifier_freezes_or_fine_tunes(is_encoder_frozen: bool, tmp_path: Path) -> None:
    X, y = make_frequencies(48, seed=0)
    learner = fit_learner(TS_TCC, X)
    path = tmp_path / "ts_tcc.pt"
    learner.save(path)
    pretrained: dict[str, Tensor] = {name: tensor.clone() for name, tensor in learner.encoder.state_dict().items()}
    model = PretrainedEncoderParameters(checkpoint_path=str(path), is_encoder_frozen=is_encoder_frozen)
    classifier = ClassifierSettings(model=model, training=training(3)).build()
    _ = classifier.fit(X, y)
    tuned: dict[str, Tensor] = encoder_state(classifier)
    is_unchanged: bool = all(torch.equal(tuned[name], tensor) for name, tensor in pretrained.items())
    assert is_unchanged == is_encoder_frozen
    trainable = [parameter for parameter in classifier.network.parameters() if parameter.requires_grad]
    num_head_parameters: int = (learner.representation_size + 1) * 2
    assert (sum(parameter.numel() for parameter in trainable) == num_head_parameters) == is_encoder_frozen
    assert classifier.predict_proba(X).values.shape == (48, 2)


def test_linear_probe_uses_the_learner_representation(tmp_path: Path) -> None:
    X, _ = make_frequencies(8, seed=0)
    learner = fit_learner(TS2VEC, X)
    learner.save(tmp_path / "ts2vec.pt")
    model = PretrainedEncoderParameters(checkpoint_path=str(tmp_path / "ts2vec.pt"), is_encoder_frozen=True)
    network = model.build_summary_network(SummaryShape(input_length=32, input_channels=1, output_size=2))
    assert isinstance(network, EncodedSummaryNetwork)
    assert network.pooling == TemporalPooling.MAX
    encoder = network.encoder
    assert isinstance(encoder, FrozenSequenceEncoder)
    _ = network.train()
    assert not encoder.encoder.training
    pretrained: dict[str, Tensor] = learner.encoder.state_dict()
    assert all(torch.equal(tensor, pretrained[name]) for name, tensor in encoder.encoder.state_dict().items())


def test_transfer_to_other_lengths_regression_and_forecasting(tmp_path: Path) -> None:
    X, _ = make_frequencies(24, seed=0)
    path = tmp_path / "ts2vec.pt"
    fit_learner(TS2VEC, X).save(path)
    model = PretrainedEncoderParameters(checkpoint_path=str(path), head=EncoderHead(pooling=TemporalPooling.MEAN))
    regressor = RegressorSettings(model=model, training=training()).build()
    _ = regressor.fit(X[:, :20], X[:, -1, :])
    assert regressor.predict(X[:, :20]).shape == (24, 1)
    forecaster = ForecasterSettings(
        model=model, window=ForecastWindow(input_length=16, horizon=4), training=training()
    ).build()
    _ = forecaster.fit(X[0])
    assert forecaster.forecast(X[0]).point.shape == (1, 4, 1)


def test_transfer_rejects_mismatched_inputs(tmp_path: Path) -> None:
    X, _ = make_frequencies(24, seed=0)
    path = tmp_path / "tst.pt"
    fit_learner(MaskedReconstructionParameters(encoder=TST), X).save(path)
    model = PretrainedEncoderParameters(checkpoint_path=str(path))
    with pytest.raises(ValueError, match="depend on the length"):
        _ = ClassifierSettings(model=model, training=training()).build().fit(X[:, :16], np.arange(24) % 2)
    with pytest.raises(ValueError, match="1 channels, got 2"):
        _ = model.build_encoder(32, 2)
    with pytest.raises(FileNotFoundError):
        _ = PretrainedEncoderParameters(checkpoint_path=str(tmp_path / "missing.pt")).build_encoder(32, 1)
    with pytest.raises(ValueError, match="checkpoint_path"):
        PretrainedEncoderParameters(checkpoint_path=" ")


@pytest.mark.skipif(
    not IS_PRETRAINED_ENCODER_REGISTERED, reason="PretrainedEncoderParameters is not a ModelArchitecture yet."
)
def test_classifier_with_pretrained_encoder_saves_and_loads(tmp_path: Path) -> None:
    X, y = make_frequencies(24, seed=0)
    fit_learner(TS2VEC, X).save(tmp_path / "ts2vec.pt")
    model = PretrainedEncoderParameters(checkpoint_path=str(tmp_path / "ts2vec.pt"), is_encoder_frozen=True)
    classifier = ClassifierSettings(model=model, training=training()).build()
    _ = classifier.fit(X, y)
    classifier.save(tmp_path / "classifier.pt")
    restored = SequenceClassifier.load(tmp_path / "classifier.pt", device=Device.CPU)
    assert np.allclose(restored.predict_proba(X).values, classifier.predict_proba(X).values, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ts2vec", TS2VecParameters()),
        ("ts_tcc", TSTCCParameters()),
        ("pretrained_encoder", PretrainedEncoderParameters(checkpoint_path="representation_learner.pt")),
    ],
)
def test_presets_hold_the_defaults(name: str, expected: object) -> None:
    document = cast(dict[str, dict[str, object]], yaml.safe_load((PRESET_DIRECTORY / f"{name}.yaml").read_text()))
    section: dict[str, object] = document["model"]
    assert section["architecture"] == name
    parameters = cast(dict[str, object], section["parameters"])
    assert ParametersReader.read(type(expected), parameters) == expected
