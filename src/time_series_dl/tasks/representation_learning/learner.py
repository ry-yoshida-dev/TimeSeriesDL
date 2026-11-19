from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ...array_types import FloatArray
from ...data import ChannelScaler, SequenceArray, SeriesCollection, SlidingWindows
from ...models import SequenceEncoder, SequenceShape
from ...models.architectures.convolutional.ts2vec import TS2VecParameters
from ...models.architectures.convolutional.ts_tcc import TSTCCParameters
from ...models.architectures.pretrained import (
    MaskedReconstructionParameters,
    PretrainedEncoderParameters,
    SimMTMParameters,
)
from ...models.components.masked_modeling import GeometricMasking, PatchMasking
from ...models.interfaces.pretraining_network import PretrainingNetwork
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .pretraining_step import PretrainingStep
from .settings import RepresentationLearnerSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    RepresentationLearnerSettings,
    TS2VecParameters,
    TSTCCParameters,
    SimMTMParameters,
    MaskedReconstructionParameters,
    PretrainedEncoderParameters,
    GeometricMasking,
    PatchMasking,
)


class RepresentationLearner:
    """
    Pretrains an encoder on unlabeled sequences with a self-supervised method and computes their representations.

    Sequences are scaled per channel with statistics of the training sequences. The pretrained encoder is
    transferred to other tasks through ``PretrainedEncoderParameters``, which reads the file written by ``save``.

    Parameters
    ----------
    settings : RepresentationLearnerSettings
        The method and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels.
    """

    def __init__(self, settings: RepresentationLearnerSettings) -> None:
        self.settings: RepresentationLearnerSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(settings.scaling)
        self._trainer: Trainer = Trainer(settings.training)
        self._network: PretrainingNetwork | None = None
        self._history: TrainingHistory | None = None
        self._length: int = 0
        self._channels: int = 0

    @property
    def network(self) -> PretrainingNetwork:
        """
        Get the pretrained network.

        Returns
        -------
        PretrainingNetwork
            The network, including the pretraining-only layers.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("RepresentationLearner is not fitted yet. Call fit() first.")
        return self._network

    @property
    def encoder(self) -> SequenceEncoder:
        """
        Get the pretrained encoder.

        Returns
        -------
        SequenceEncoder
            The encoder.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        return self.network.encoder

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
            raise NotFittedError("RepresentationLearner is not fitted yet. Call fit() first.")
        return self._history

    @property
    def representation_size(self) -> int:
        """
        Get the size of the instance-level representations.

        Returns
        -------
        int
            The number of columns of ``encode``.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        return self.network.representation_size

    def fit(self, X: FloatArray, validation: FloatArray | None = None) -> TrainingHistory:
        """
        Pretrain a new encoder on unlabeled sequences.

        Parameters
        ----------
        X : FloatArray
            Training sequences with shape (num_sequences, length, channels).
        validation : FloatArray | None
            Sequences whose loss selects the best epoch and stops training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the sequences are not numeric.
        ValueError
            If the shapes are invalid, or the method cannot pretrain on sequences of this length.
        """
        sequences: SequenceArray = SequenceArray(X)
        validation_sequences: SequenceArray | None = None
        if validation is not None:
            validation_sequences = SequenceArray(validation)
            validation_sequences.require_shape(sequences.length, sequences.channels)
        return self._fit(sequences.values, None if validation_sequences is None else validation_sequences.values)

    def fit_series(
        self,
        series: FloatArray | Sequence[FloatArray],
        window_length: int,
        stride: int = 1,
        validation_series: FloatArray | Sequence[FloatArray] | None = None,
    ) -> TrainingHistory:
        """
        Pretrain a new encoder on the sliding windows of unlabeled series.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one window long.
        window_length : int
            Number of time steps of each window; the encoder reads sequences of this length.
        stride : int
            Number of steps between the starts of consecutive windows.
        validation_series : FloatArray | Sequence[FloatArray] | None
            Series whose windows select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If a series is invalid or shorter than one window, the channels differ, or window_length or stride is
            not positive.
        """
        windows: SlidingWindows = SlidingWindows(window_length, stride)
        training: FloatArray = self._extract_windows(SeriesCollection.from_input(series), windows)
        validation: FloatArray | None = None
        if validation_series is not None:
            validation = self._extract_windows(SeriesCollection.from_input(validation_series), windows)
            if validation.shape[2] != training.shape[2]:
                raise ValueError(f"validation series must have {training.shape[2]} channels, got {validation.shape[2]}")
        return self._fit(training, validation)

    def encode(self, X: FloatArray) -> FloatArray:
        """
        Compute the instance-level representation of each sequence.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).

        Returns
        -------
        FloatArray
            Representations with shape (num_sequences, representation_size), pooled from the encoder's features as
            the method's paper does.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid or have another length or number of channels.
        """
        network: PretrainingNetwork = self.network
        return self._run(network.encode, X)

    def encode_timestamps(self, X: FloatArray) -> FloatArray:
        """
        Compute the timestamp-level representations of each sequence.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).

        Returns
        -------
        FloatArray
            Representations with shape (num_sequences, encoder.output_length, encoder.output_size); the output
            length equals the input length for encoders that keep it, such as those of TS2Vec and TST.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid or have another length or number of channels.
        """
        encoder: SequenceEncoder = self.encoder
        return self._run(encoder, X)

    def build_transfer_network(self, length: int, channels: int) -> PretrainingNetwork:
        """
        Build a new pretraining network for sequences of another length, whose encoder has the pretrained weights.

        Parameters
        ----------
        length : int
            Number of time steps of the sequences of the task the encoder is transferred to.
        channels : int
            Number of channels of those sequences; must be the number of channels of pretraining.

        Returns
        -------
        PretrainingNetwork
            The network on the CPU; only its encoder is pretrained.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the number of channels differs, or the encoder's weights depend on the length and it differs.
        """
        pretrained: dict[str, Tensor] = self.encoder.state_dict()
        if channels != self._channels:
            raise ValueError(f"the encoder was pretrained on {self._channels} channels, got {channels}")
        network: PretrainingNetwork = self.settings.method.build_pretraining_network(
            SequenceShape(length=length, channels=channels)
        )
        expected: dict[str, Tensor] = network.encoder.state_dict()
        mismatched: list[str] = sorted(
            name
            for name in set(expected) | set(pretrained)
            if name not in expected or name not in pretrained or expected[name].shape != pretrained[name].shape
        )
        if mismatched:
            raise ValueError(
                f"the encoder pretrained on sequences of length {self._length} cannot read length {length}: "
                + f"weights {mismatched} depend on the length"
            )
        _ = network.encoder.load_state_dict(pretrained)
        return network.eval()

    def save(self, path: Path) -> None:
        """
        Write the fitted learner to a file.

        The file holds the settings, the scaling statistics, the weights of the whole pretraining network and the
        training history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``RepresentationLearner.load`` and ``PretrainedEncoderParameters`` read it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: PretrainingNetwork = self.network
        Checkpoint(
            task=TaskKind.REPRESENTATION_LEARNING,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={"length": self._length, "channels": self._channels},
            scalers={"input": self._scaler.to_state()},
            network=network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a learner written by save, ready to encode without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted learner.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a representation learner checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.REPRESENTATION_LEARNING)
        settings: RepresentationLearnerSettings = _CODEC.decode(RepresentationLearnerSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        learner: Self = cls(settings)
        learner._length = checkpoint.read_int("length")
        learner._channels = checkpoint.read_int("channels")
        learner._scaler.restore_state(checkpoint.read_scaler("input"), learner._channels)
        network: PretrainingNetwork = learner._build_network()
        checkpoint.restore_network(network, learner._trainer.device)
        learner._network = network
        learner._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return learner

    def _fit(self, training: FloatArray, validation: FloatArray | None) -> TrainingHistory:
        self._network = None
        self._history = None
        self._scaler.fit(training)
        self._length = int(training.shape[1])
        self._channels = int(training.shape[2])
        self._trainer.seed()
        network: PretrainingNetwork = self._build_network()
        history: TrainingHistory = self._trainer.fit(
            network,
            PretrainingStep(network),
            self._build_dataset(training),
            None if validation is None else self._build_dataset(validation),
        )
        self._network = network
        self._history = history
        return history

    def _build_network(self) -> PretrainingNetwork:
        return self.settings.method.build_pretraining_network(
            SequenceShape(length=self._length, channels=self._channels)
        )

    def _build_dataset(self, sequences: FloatArray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(self._scaler.transform(sequences)))

    def _run(self, function: Callable[[Tensor], Tensor], X: FloatArray) -> FloatArray:
        sequences: SequenceArray = SequenceArray(X)
        sequences.require_shape(self._length, self._channels)
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: function(batch[0]), TensorDataset(torch.from_numpy(self._scaler.transform(sequences.values)))
        )
        return outputs.numpy()

    @staticmethod
    def _extract_windows(collection: SeriesCollection, windows: SlidingWindows) -> FloatArray:
        if collection.shortest_length < windows.length:
            raise ValueError(
                f"every series must have at least {windows.length} steps, got {collection.shortest_length}"
            )
        return np.concatenate([windows.extract(values) for values in collection.series], axis=0)
