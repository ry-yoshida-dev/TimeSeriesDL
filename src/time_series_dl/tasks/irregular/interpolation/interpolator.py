from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ....array_types import BoolArray, FloatArray, IndexArray
from ....data.irregular import IrregularScaler, IrregularSequenceCollection, PaddedIrregularSequences, QueryTimes
from ....models.architectures.continuous_time import LatentODEParameters, MTANParameters
from ....models.interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....models.interfaces.irregular_tensors import IrregularTensors
from ....models.interfaces.protocols.irregular_interpolation import IrregularInterpolationModel
from ....models.interfaces.shapes.irregular_sequence import IrregularSequenceShape
from ....persistence import Checkpoint, SettingsCodec, TaskKind, restorable_model
from ....runtime import BatchedInference, Device
from ....training import Trainer, TrainingHistory
from .interpolation_step import IrregularInterpolationStep
from .metrics import InterpolationMetrics
from .settings import IrregularInterpolatorSettings

# The irregular models are listed until ModelArchitecture names them, which adds them to every codec.
_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    IrregularInterpolatorSettings, MTANParameters, LatentODEParameters
)


class IrregularInterpolator:
    """
    Predicts the values of irregularly sampled sequences at any times from their observed values: between the
    observations (interpolation) or after them (forecasting).

    Values are scaled per channel with statistics of the observed training values, and times so that the training
    times span [0, 1]; predictions are returned in the original scale.

    Parameters
    ----------
    settings : IrregularInterpolatorSettings
        The model, context selection and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels or is not invertible.
    """

    def __init__(self, settings: IrregularInterpolatorSettings) -> None:
        self.settings: IrregularInterpolatorSettings = settings
        self._scaler: IrregularScaler = IrregularScaler(settings.scaling)
        if not self._scaler.is_invertible:
            raise ValueError(f"{type(settings.scaling).__name__} has no inverse; predictions could not be unscaled.")
        self._trainer: Trainer = Trainer(settings.training)
        self._network: IrregularInterpolationNetwork | None = None
        self._history: TrainingHistory | None = None
        self._channels: int = 0

    @property
    def network(self) -> IrregularInterpolationNetwork:
        """
        Get the trained network.

        Returns
        -------
        IrregularInterpolationNetwork
            The network predicting the scaled values.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("IrregularInterpolator is not fitted yet. Call fit() first.")
        return self._network

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
            raise NotFittedError("IrregularInterpolator is not fitted yet. Call fit() first.")
        return self._history

    def fit(
        self, sequences: IrregularSequenceCollection, validation: IrregularSequenceCollection | None = None
    ) -> TrainingHistory:
        """
        Train a new network to predict hidden observations from the others.

        Parameters
        ----------
        sequences : IrregularSequenceCollection
            Training sequences.
        validation : IrregularSequenceCollection | None
            Sequences that select the best epoch and stop training early, split with a fixed random selection.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If a channel is never observed or the validation sequences have another number of channels.
        """
        if validation is not None:
            validation.require_channels(sequences.channels)
        self._network = None
        self._history = None
        self._channels = sequences.channels
        self._scaler.fit(sequences.pad())
        self._trainer.seed()
        network: IrregularInterpolationNetwork = self._build_network()
        history: TrainingHistory = self._trainer.fit(
            network,
            IrregularInterpolationStep(
                network=network,
                context_selection=self.settings.context_selection,
                context_ratio=self.settings.context_ratio,
            ),
            self._build_dataset(sequences),
            None if validation is None else self._build_dataset(validation),
        )
        self._network = network
        self._history = history
        return history

    def predict(self, context: IrregularSequenceCollection, query_times: QueryTimes) -> list[FloatArray]:
        """
        Predict the values of each sequence at its query times.

        Parameters
        ----------
        context : IrregularSequenceCollection
            The observed sequences.
        query_times : QueryTimes
            The times to predict, one array per sequence; they may lie between or after the observations.

        Returns
        -------
        list[FloatArray]
            The predicted values of each sequence with shape (num_queries, channels), in the original scale.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the number of sequences or of channels differs.
        """
        network: IrregularInterpolationNetwork = self.network
        context.require_channels(self._channels)
        if query_times.num_sequences != context.num_sequences:
            raise ValueError(
                f"query_times must cover {context.num_sequences} sequences, got {query_times.num_sequences}"
            )
        padded: PaddedIrregularSequences = self._scaler.transform(context.pad())
        queries: Tensor = torch.from_numpy(self._scaler.transform_times(query_times.pad()).astype(np.float32))
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: network(IrregularTensors.from_batch(batch).trimmed(), batch[4]),
            TensorDataset(*padded.to_tensors(), queries),
        )
        values: FloatArray = self._scaler.inverse_transform_values(outputs.double().numpy())
        lengths: IndexArray = query_times.lengths
        return [values[index, : lengths[index]] for index in range(context.num_sequences)]

    def evaluate(
        self, context: IrregularSequenceCollection, targets: IrregularSequenceCollection
    ) -> InterpolationMetrics:
        """
        Measure the predictions at the times of known values.

        Parameters
        ----------
        context : IrregularSequenceCollection
            The observed sequences.
        targets : IrregularSequenceCollection
            The values to predict, one sequence per context sequence; only observed values are compared.

        Returns
        -------
        InterpolationMetrics
            The errors in the original scale.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the number of sequences or of channels differs.
        """
        targets.require_channels(self._channels)
        predictions: list[FloatArray] = self.predict(
            context, QueryTimes(tuple(sequence.times for sequence in targets.sequences))
        )
        return InterpolationMetrics.compute(predictions, targets)

    def save(self, path: Path) -> None:
        """
        Write the fitted interpolator to a file.

        The file holds the settings, the scaling statistics and time range, the network weights and the training
        history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``IrregularInterpolator.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: IrregularInterpolationNetwork = self.network
        Checkpoint(
            task=TaskKind.IRREGULAR_INTERPOLATION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={
                "channels": self._channels,
                "time_offset": self._scaler.time_offset,
                "time_span": self._scaler.time_span,
            },
            scalers={"input": self._scaler.to_state()},
            network=network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read an interpolator written by save, ready to predict without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted interpolator.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not an irregular interpolation checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.IRREGULAR_INTERPOLATION)
        settings: IrregularInterpolatorSettings = _CODEC.decode(IrregularInterpolatorSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        interpolator: Self = cls(settings)
        interpolator._channels = checkpoint.read_int("channels")
        interpolator._scaler.restore_state(
            checkpoint.read_scaler("input"),
            interpolator._channels,
            checkpoint.read_float("time_offset"),
            checkpoint.read_float("time_span"),
        )
        network: IrregularInterpolationNetwork = interpolator._build_network(is_restoring=True)
        checkpoint.restore_network(network, interpolator._trainer.device)
        interpolator._network = network
        interpolator._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return interpolator

    def _build_network(self, is_restoring: bool = False) -> IrregularInterpolationNetwork:
        model: IrregularInterpolationModel = (
            restorable_model(self.settings.model) if is_restoring else self.settings.model
        )
        return model.build_irregular_interpolation_network(IrregularSequenceShape(channels=self._channels))

    def _build_dataset(self, sequences: IrregularSequenceCollection) -> TensorDataset:
        padded: PaddedIrregularSequences = self._scaler.transform(sequences.pad())
        seed: int | None = self.settings.training.random_seed
        is_drawn: BoolArray = np.random.default_rng(seed).random(padded.values.shape) < self.settings.context_ratio
        return TensorDataset(*padded.to_tensors(), torch.from_numpy(is_drawn))
