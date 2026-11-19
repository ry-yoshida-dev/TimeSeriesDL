from pathlib import Path

import numpy as np
import pytest
from deep_learning_loss import FocalLoss
from irregular_datasets import make_timing_records
from sklearn.exceptions import NotFittedError

from time_series_dl import Device, TrainingSettings
from time_series_dl.data.irregular import IrregularSequenceCollection
from time_series_dl.models.architectures.continuous_time import (
    LatentODEParameters,
    MTANParameters,
    NeuralCDEParameters,
    ODERNNParameters,
)
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.models.interfaces.protocols.irregular_summary import IrregularSummaryModel
from time_series_dl.tasks.irregular.classification import IrregularClassifierSettings, IrregularSequenceClassifier

SMALL_HEAD: IrregularEncoderHead = IrregularEncoderHead(hidden_sizes=(16,))
LEARNING_MODELS: list[IrregularSummaryModel] = [
    MTANParameters(hidden_size=16, time_embedding_size=8, num_reference_points=16, head=SMALL_HEAD),
    ODERNNParameters(hidden_size=16, ode_hidden_sizes=(16,), head=SMALL_HEAD),
    LatentODEParameters(
        latent_size=4,
        recognition_size=8,
        recognition_hidden_sizes=(16,),
        dynamics_hidden_sizes=(16,),
        observation_std=0.1,
        output_loss_weight=10.0,
        head=SMALL_HEAD,
    ),
]


@pytest.mark.parametrize("model", LEARNING_MODELS, ids=lambda model: type(model).__name__)
def test_classifier_learns_when_the_step_happens(model: IrregularSummaryModel) -> None:
    X, y = make_timing_records(200, seed=0)
    X_test, y_test = make_timing_records(100, seed=1)
    classifier = IrregularClassifierSettings(
        model=model,
        training=TrainingSettings(max_epochs=30, learning_rate=1e-2, device=Device.CPU, early_stopping_patience=None),
    ).build()
    history = classifier.fit(X, y)
    assert history.num_epochs == 30
    metrics = classifier.evaluate(X_test, y_test)
    assert metrics.accuracy > 0.85
    assert metrics.roc_auc is not None


def test_classifier_round_trips_and_validates(tmp_path: Path, fast_training: TrainingSettings) -> None:
    X, y = make_timing_records(40, seed=2)
    labels: np.ndarray = np.where(y == 1, "early", "late")
    classifier = IrregularClassifierSettings(
        model=NeuralCDEParameters(hidden_size=4, field_hidden_size=8, field_layers=1),
        training=fast_training,
        loss=FocalLoss(gamma=2.0),
        is_class_balanced=True,
    ).build()
    with pytest.raises(NotFittedError):
        _ = classifier.predict(X)
    _ = classifier.fit(X, labels, validation=(X, labels))
    assert classifier.classes.tolist() == ["early", "late"]
    probabilities: np.ndarray = classifier.predict_proba(X).values
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    path: Path = tmp_path / "classifier.pt"
    classifier.save(path)
    restored = IrregularSequenceClassifier.load(path, device=Device.CPU)
    assert np.allclose(restored.predict_proba(X).values, probabilities, atol=1e-6)
    assert restored.settings == classifier.settings
    assert restored.history == classifier.history
    with pytest.raises(ValueError, match="at least 2 classes"):
        _ = classifier.fit(X, np.zeros(40, dtype=np.int64))
    with pytest.raises(ValueError, match="not fitted on"):
        _ = classifier.evaluate(X, np.full(40, "other"))
    one_channel = IrregularSequenceCollection.from_arrays([np.array([0.0])], [np.array([[1.0]])])
    with pytest.raises(ValueError, match="2 channels"):
        _ = classifier.predict(one_channel)
