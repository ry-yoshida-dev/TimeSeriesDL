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

from ....array_types import FloatArray, IndexArray, LabelArray
from ....data.irregular import IrregularScaler, IrregularSequenceCollection
from ....models.architectures.continuous_time import (
    LatentODEParameters,
    MTANParameters,
    NeuralCDEParameters,
    ODERNNParameters,
)
from ....models.interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....models.interfaces.irregular_tensors import IrregularTensors
from ....models.interfaces.shapes.irregular_summary import IrregularSummaryShape
from ....persistence import Checkpoint, EncodedValue, SettingsCodec, TaskKind
from ....runtime import BatchedInference, Device
from ....training import Trainer, TrainingHistory
from ...classification import CrossEntropyLoss, FocalLoss
from .classification_step import IrregularClassificationStep
from .settings import IrregularClassifierSettings

# The irregular models are listed until ModelArchitecture names them, which adds them to every codec.
_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    IrregularClassifierSettings,
    CrossEntropyLoss,
    FocalLoss,
    MTANParameters,
    NeuralCDEParameters,
    ODERNNParameters,
    LatentODEParameters,
)
_LABEL_KINDS: Final[str] = "iubU"


class IrregularSequenceClassifier:
    """
    Assigns one class to each irregularly sampled multivariate sequence, such as in-hospital mortality from the ICU
    records of PhysioNet.

    Values are scaled per channel with statistics of the observed training values, and times so that the training
    times span [0, 1]. Metrics are those of ``classreg.classification``, as for ``SequenceClassifier``.

    Parameters
    ----------
    settings : IrregularClassifierSettings
        The model and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels.
    """

    def __init__(self, settings: IrregularClassifierSettings) -> None:
        self.settings: IrregularClassifierSettings = settings
        self._scaler: IrregularScaler = IrregularScaler(settings.scaling)
        self._trainer: Trainer = Trainer(settings.training)
        self._network: IrregularSummaryNetwork | None = None
        self._classes: LabelArray | None = None
        self._history: TrainingHistory | None = None
        self._channels: int = 0

    @property
    def network(self) -> IrregularSummaryNetwork:
        """
        Get the trained network.

        Returns
        -------
        IrregularSummaryNetwork
            The network producing one logit per class.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("IrregularSequenceClassifier is not fitted yet. Call fit() first.")
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
            raise NotFittedError("IrregularSequenceClassifier is not fitted yet. Call fit() first.")
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
            raise NotFittedError("IrregularSequenceClassifier is not fitted yet. Call fit() first.")
        return self._history

    def fit(
        self,
        X: IrregularSequenceCollection,
        y: LabelArray,
        validation: tuple[IrregularSequenceCollection, LabelArray] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Training sequences.
        y : LabelArray
            Integer, boolean or string class labels with shape (num_sequences,).
        validation : tuple[IrregularSequenceCollection, LabelArray] | None
            Sequences and labels that select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the labels are not discrete.
        ValueError
            If the shapes are invalid, a channel is never observed, there are fewer than two classes, or validation
            labels are unseen.
        """
        labels: LabelArray = self._validate_labels(y, X.num_sequences)
        classes: LabelArray = np.unique(labels)
        if classes.size < 2:
            raise ValueError(f"y must contain at least 2 classes, got {classes.size}")
        validation_data: tuple[IrregularSequenceCollection, IndexArray] | None = None
        if validation is not None:
            validation[0].require_channels(X.channels)
            validation_labels: LabelArray = self._validate_labels(validation[1], validation[0].num_sequences)
            validation_data = (validation[0], self._encode(validation_labels, classes))
        self._network = None
        self._classes = None
        self._history = None
        self._scaler.fit(X.pad())
        self._channels = X.channels
        indices: IndexArray = self._encode(labels, classes)
        self._trainer.seed()
        network: IrregularSummaryNetwork = self._build_network(classes.size)
        step: IrregularClassificationStep = IrregularClassificationStep(
            network=network,
            class_weights=self._compute_class_weights(indices, classes.size),
            loss=self.settings.loss,
        )
        history: TrainingHistory = self._trainer.fit(
            network,
            step,
            self._build_dataset(X, indices),
            None if validation_data is None else self._build_dataset(*validation_data),
        )
        classes.setflags(write=False)
        self._network = network
        self._classes = classes
        self._history = history
        return history

    def predict_proba(self, X: IrregularSequenceCollection) -> ClassScores:
        """
        Estimate class membership probabilities.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Sequences.

        Returns
        -------
        ClassScores
            The probabilities with shape (num_sequences, num_classes), columns in the order of ``classes``.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences have another number of channels.
        """
        network: IrregularSummaryNetwork = self.network
        X.require_channels(self._channels)
        logits: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: network(IrregularTensors.from_batch(batch).trimmed()),
            TensorDataset(*self._scaler.transform(X.pad()).to_tensors()),
        )
        probabilities: FloatArray = torch.softmax(logits.double(), dim=1).numpy()
        return ClassScores(classes=self.classes, values=probabilities, kind=ClassScoreKind.PROBABILITY)

    def predict(self, X: IrregularSequenceCollection) -> LabelArray:
        """
        Predict the most probable class of each sequence.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Sequences.

        Returns
        -------
        LabelArray
            The predicted labels with shape (num_sequences,).

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences have another number of channels.
        """
        scores: ClassScores = self.predict_proba(X)
        return self.classes[np.argmax(scores.values, axis=1)]

    def evaluate(self, X: IrregularSequenceCollection, y: LabelArray) -> MetricValues:
        """
        Measure the fitted classifier on labeled sequences.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Sequences.
        y : LabelArray
            The true class labels.

        Returns
        -------
        MetricValues
            The ``classreg`` classification metrics, including the AUROC and AUPRC of binary labels.

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

        The file holds the settings, the classes, the scaling statistics and time range, the network weights and the
        training history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``IrregularSequenceClassifier.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: IrregularSummaryNetwork = self.network
        classes: LabelArray = self.classes
        Checkpoint(
            task=TaskKind.IRREGULAR_CLASSIFICATION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={
                "channels": self._channels,
                "classes": cast(list[EncodedValue], classes.tolist()),
                "class_dtype": classes.dtype.str,
                "time_offset": self._scaler.time_offset,
                "time_span": self._scaler.time_span,
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
            If the file is not an irregular classifier checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.IRREGULAR_CLASSIFICATION)
        settings: IrregularClassifierSettings = _CODEC.decode(IrregularClassifierSettings, checkpoint.settings)
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
        classifier._channels = checkpoint.read_int("channels")
        classifier._scaler.restore_state(
            checkpoint.read_scaler("input"),
            classifier._channels,
            checkpoint.read_float("time_offset"),
            checkpoint.read_float("time_span"),
        )
        network: IrregularSummaryNetwork = classifier._build_network(classes.size)
        checkpoint.restore_network(network, classifier._trainer.device)
        classifier._network = network
        classifier._classes = classes
        classifier._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return classifier

    def _build_network(self, num_classes: int) -> IrregularSummaryNetwork:
        return self.settings.model.build_irregular_summary_network(
            IrregularSummaryShape(input_channels=self._channels, output_size=num_classes)
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

    def _build_dataset(self, sequences: IrregularSequenceCollection, indices: IndexArray) -> TensorDataset:
        return TensorDataset(
            *self._scaler.transform(sequences.pad()).to_tensors(), torch.from_numpy(indices.astype(np.int64))
        )
