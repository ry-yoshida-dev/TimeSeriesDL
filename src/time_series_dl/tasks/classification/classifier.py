from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final, Self, cast

import numpy as np
import torch
from classreg.classification import ClassScoreKind, ClassScores, MetricValues
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ...array_types import FloatArray, IndexArray, LabelArray
from ...data import ChannelScaler, SequenceArray
from ...models import SummaryNetwork, SummaryShape
from ...persistence import Checkpoint, EncodedValue, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .classification_step import ClassificationStep
from .settings import ClassifierSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(ClassifierSettings)
_LABEL_KINDS: Final[str] = "iubU"


class SequenceClassifier:
    """
    Assigns one class to each multivariate sequence.

    Sequences are scaled per channel with statistics of the training sequences. Metrics are those of
    ``classreg.classification``, so deep and classical classifiers can be compared directly.

    Parameters
    ----------
    settings : ClassifierSettings
        The model and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels.
    """

    def __init__(self, settings: ClassifierSettings) -> None:
        self.settings: ClassifierSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(settings.scaling)
        self._trainer: Trainer = Trainer(settings.training)
        self._network: SummaryNetwork | None = None
        self._classes: LabelArray | None = None
        self._history: TrainingHistory | None = None
        self._length: int = 0
        self._channels: int = 0

    @property
    def network(self) -> SummaryNetwork:
        """
        Get the trained network.

        Returns
        -------
        SummaryNetwork
            The network producing one logit per class.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("SequenceClassifier is not fitted yet. Call fit() first.")
        return self._network

    @property
    def classes(self) -> LabelArray:
        """
        Get the sorted class labels seen during fit.

        Returns
        -------
        LabelArray
            The read-only class labels.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._classes is None:
            raise NotFittedError("SequenceClassifier is not fitted yet. Call fit() first.")
        return self._classes

    @property
    def history(self) -> TrainingHistory:
        """
        Get the losses of the last training run.

        Returns
        -------
        TrainingHistory
            The history.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._history is None:
            raise NotFittedError("SequenceClassifier is not fitted yet. Call fit() first.")
        return self._history

    def fit(
        self, X: FloatArray, y: LabelArray, validation: tuple[FloatArray, LabelArray] | None = None
    ) -> TrainingHistory:
        """
        Train a new network.

        Parameters
        ----------
        X : FloatArray
            Training sequences with shape (num_sequences, length, channels).
        y : LabelArray
            Integer, boolean or string class labels with shape (num_sequences,).
        validation : tuple[FloatArray, LabelArray] | None
            Sequences and labels that select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the sequences are not numeric or the labels are not discrete.
        ValueError
            If the shapes are invalid, there are fewer than two classes, or validation labels are unseen.
        """
        sequences: SequenceArray = SequenceArray(X)
        labels: LabelArray = self._validate_labels(y, sequences.num_sequences)
        classes: LabelArray = np.unique(labels)
        if classes.size < 2:
            raise ValueError(f"y must contain at least 2 classes, got {classes.size}")
        validation_data: tuple[SequenceArray, IndexArray] | None = None
        if validation is not None:
            validation_sequences: SequenceArray = SequenceArray(validation[0])
            validation_sequences.require_shape(sequences.length, sequences.channels)
            validation_labels: LabelArray = self._validate_labels(validation[1], validation_sequences.num_sequences)
            validation_data = (validation_sequences, self._encode(validation_labels, classes))
        self._network = None
        self._classes = None
        self._history = None
        self._scaler.fit(sequences.values)
        self._length = sequences.length
        self._channels = sequences.channels
        indices: IndexArray = self._encode(labels, classes)
        self._trainer.seed()
        network: SummaryNetwork = self._build_network(classes.size)
        step: ClassificationStep = ClassificationStep(
            network=network,
            class_weights=self._compute_class_weights(indices, classes.size),
            loss=self.settings.loss,
        )
        history: TrainingHistory = self._trainer.fit(
            network,
            step,
            self._build_dataset(sequences, indices),
            None if validation_data is None else self._build_dataset(*validation_data),
        )
        classes.setflags(write=False)
        self._network = network
        self._classes = classes
        self._history = history
        return history

    def predict_proba(self, X: FloatArray) -> ClassScores:
        """
        Estimate class membership probabilities.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).

        Returns
        -------
        ClassScores
            The probabilities with shape (num_sequences, num_classes), columns in the order of ``classes``.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid or have another length or number of channels.
        """
        network: SummaryNetwork = self.network
        sequences: SequenceArray = SequenceArray(X)
        sequences.require_shape(self._length, self._channels)
        logits: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: network(batch[0]), TensorDataset(torch.from_numpy(self._scaler.transform(sequences.values)))
        )
        probabilities: FloatArray = torch.softmax(logits.double(), dim=1).numpy()
        return ClassScores(classes=self.classes, values=probabilities, kind=ClassScoreKind.PROBABILITY)

    def predict(self, X: FloatArray) -> LabelArray:
        """
        Predict the most probable class of each sequence.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).

        Returns
        -------
        LabelArray
            The predicted labels with shape (num_sequences,).

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid or have another length or number of channels.
        """
        scores: ClassScores = self.predict_proba(X)
        return self.classes[np.argmax(scores.values, axis=1)]

    def evaluate(self, X: FloatArray, y: LabelArray) -> MetricValues:
        """
        Measure the fitted classifier on labeled sequences.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).
        y : LabelArray
            The true class labels.

        Returns
        -------
        MetricValues
            The ``classreg`` classification metrics.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the inputs are invalid or y contains labels unseen during fit.
        """
        scores: ClassScores = self.predict_proba(X)
        labels: LabelArray = self._validate_labels(y, scores.values.shape[0])
        predicted: LabelArray = self.classes[np.argmax(scores.values, axis=1)]
        return MetricValues.compute(y_true=labels, y_pred=predicted, scores=scores)

    def save(self, path: Path) -> None:
        """
        Write the fitted classifier to a file.

        The file holds the settings, the classes, the scaling statistics, the network weights and the training
        history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``SequenceClassifier.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: SummaryNetwork = self.network
        classes: LabelArray = self.classes
        Checkpoint(
            task=TaskKind.CLASSIFICATION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={
                "length": self._length,
                "channels": self._channels,
                "classes": cast(list[EncodedValue], classes.tolist()),
                "class_dtype": classes.dtype.str,
            },
            scalers={"input": self._scaler.to_state()},
            network=network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a classifier written by save, ready to predict without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted classifier.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a classifier checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.CLASSIFICATION)
        settings: ClassifierSettings = _CODEC.decode(ClassifierSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        classifier: Self = cls(settings)
        class_dtype: np.dtype[np.generic] = np.dtype(checkpoint.read_str("class_dtype"))
        if class_dtype.kind not in _LABEL_KINDS:
            raise ValueError(f"checkpoint classes must be integers, booleans or strings, got dtype {class_dtype}")
        # The kind check above restricts the dtype to that of labels.
        classes: LabelArray = cast(LabelArray, np.asarray(checkpoint.read_list("classes"), dtype=class_dtype))
        if classes.ndim != 1 or classes.size < 2:
            raise ValueError(f"checkpoint must hold at least 2 classes, got {classes.tolist()}")
        classes.setflags(write=False)
        classifier._length = checkpoint.read_int("length")
        classifier._channels = checkpoint.read_int("channels")
        classifier._scaler.restore_state(checkpoint.read_scaler("input"), classifier._channels)
        network: SummaryNetwork = classifier._build_network(classes.size)
        checkpoint.restore_network(network, classifier._trainer.device)
        classifier._network = network
        classifier._classes = classes
        classifier._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return classifier

    def _build_network(self, num_classes: int) -> SummaryNetwork:
        return self.settings.model.build_summary_network(
            SummaryShape(input_length=self._length, input_channels=self._channels, output_size=num_classes)
        )

    def _compute_class_weights(self, indices: IndexArray, num_classes: int) -> Tensor | None:
        if not self.settings.is_class_balanced:
            return None
        counts: FloatArray = np.bincount(indices, minlength=num_classes).astype(np.float64)
        return torch.from_numpy(counts.sum() / (num_classes * counts)).float()

    @staticmethod
    def _validate_labels(y: LabelArray, num_sequences: int) -> LabelArray:
        labels: LabelArray = np.asarray(y)
        is_discrete: bool = any(np.issubdtype(labels.dtype, kind) for kind in (np.integer, np.bool_, np.str_))
        if not is_discrete:
            raise TypeError(f"y must contain integer, boolean, or string labels, got dtype {labels.dtype}")
        if labels.shape != (num_sequences,):
            raise ValueError(f"y must have shape ({num_sequences},), got {labels.shape}")
        return labels

    @staticmethod
    def _encode(labels: LabelArray, classes: LabelArray) -> IndexArray:
        unknown: LabelArray = np.setdiff1d(labels, classes)
        if unknown.size > 0:
            raise ValueError(f"labels unseen during fit: {unknown.tolist()}")
        return np.searchsorted(classes, labels).astype(np.intp)

    def _build_dataset(self, sequences: SequenceArray, indices: IndexArray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(self._scaler.transform(sequences.values)), torch.from_numpy(indices.astype(np.int64))
        )
