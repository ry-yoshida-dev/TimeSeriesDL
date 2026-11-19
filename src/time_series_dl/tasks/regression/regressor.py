from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch
from classreg.regression import MetricValues
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ...array_types import FloatArray
from ...data import ChannelScaler, SequenceArray
from ...models import SummaryNetwork, SummaryShape
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .regression_step import RegressionStep
from .settings import RegressorSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(RegressorSettings)


class SequenceRegressor:
    """
    Predicts one or more values from each multivariate sequence.

    Sequences are scaled per channel and targets per output, both with training statistics; predictions are returned
    in the original scale. Metrics are those of ``classreg.regression``, one per output.

    Parameters
    ----------
    settings : RegressorSettings
        The model, loss and training settings.

    Raises
    ------
    ValueError
        If a scaling strategy mixes channels, or the target scaling is not invertible.
    """

    def __init__(self, settings: RegressorSettings) -> None:
        self.settings: RegressorSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(settings.scaling)
        self._target_scaler: ChannelScaler = ChannelScaler(settings.target_scaling)
        if not self._target_scaler.is_invertible:
            raise ValueError(
                f"{type(settings.target_scaling).__name__} has no inverse; predictions could not be unscaled."
            )
        self._trainer: Trainer = Trainer(settings.training)
        self._network: SummaryNetwork | None = None
        self._history: TrainingHistory | None = None
        self._is_target_vector: bool = False
        self._num_outputs: int = 0
        self._length: int = 0
        self._channels: int = 0

    @property
    def network(self) -> SummaryNetwork:
        """
        Get the trained network.

        Returns
        -------
        SummaryNetwork
            The network producing the scaled targets.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("SequenceRegressor is not fitted yet. Call fit() first.")
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
            raise NotFittedError("SequenceRegressor is not fitted yet. Call fit() first.")
        return self._history

    def fit(
        self, X: FloatArray, y: FloatArray, validation: tuple[FloatArray, FloatArray] | None = None
    ) -> TrainingHistory:
        """
        Train a new network.

        Parameters
        ----------
        X : FloatArray
            Training sequences with shape (num_sequences, length, channels).
        y : FloatArray
            Targets with shape (num_sequences,) or (num_sequences, num_outputs).
        validation : tuple[FloatArray, FloatArray] | None
            Sequences and targets that select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the sequences or targets are not numeric.
        ValueError
            If the shapes are invalid or a value is not finite.
        """
        sequences: SequenceArray = SequenceArray(X)
        targets: FloatArray = self._validate_targets(y, sequences.num_sequences)
        validation_data: tuple[SequenceArray, FloatArray] | None = None
        if validation is not None:
            validation_sequences: SequenceArray = SequenceArray(validation[0])
            validation_sequences.require_shape(sequences.length, sequences.channels)
            validation_targets: FloatArray = self._validate_targets(validation[1], validation_sequences.num_sequences)
            if validation_targets.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"validation targets must have {targets.shape[1]} outputs, got {validation_targets.shape[1]}"
                )
            validation_data = (validation_sequences, validation_targets)
        self._network = None
        self._history = None
        self._is_target_vector = np.asarray(y).ndim == 2
        self._num_outputs = int(targets.shape[1])
        self._length = sequences.length
        self._channels = sequences.channels
        self._scaler.fit(sequences.values)
        self._target_scaler.fit(targets)
        self._trainer.seed()
        network: SummaryNetwork = self._build_network()
        history: TrainingHistory = self._trainer.fit(
            network,
            RegressionStep(network=network, loss=self.settings.loss),
            self._build_dataset(sequences, targets),
            None if validation_data is None else self._build_dataset(*validation_data),
        )
        self._network = network
        self._history = history
        return history

    def predict(self, X: FloatArray) -> FloatArray:
        """
        Predict the targets of each sequence.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).

        Returns
        -------
        FloatArray
            Predictions in the original scale, with shape (num_sequences,) when fitted on 1-D targets and
            (num_sequences, num_outputs) otherwise.

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
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: network(batch[0]), TensorDataset(torch.from_numpy(self._scaler.transform(sequences.values)))
        )
        predictions: FloatArray = self._target_scaler.inverse_transform(outputs.double().numpy())
        return predictions if self._is_target_vector else predictions[:, 0]

    def evaluate(self, X: FloatArray, y: FloatArray) -> tuple[MetricValues, ...]:
        """
        Measure the fitted regressor on sequences with known targets.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels).
        y : FloatArray
            The true targets, shaped like the training targets.

        Returns
        -------
        tuple[MetricValues, ...]
            The ``classreg`` regression metrics of each output.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the inputs are invalid or the number of outputs differs from fit.
        """
        predictions: FloatArray = self.predict(X).reshape(-1, self._num_outputs)
        targets: FloatArray = self._validate_targets(y, predictions.shape[0])
        if targets.shape[1] != self._num_outputs:
            raise ValueError(f"y must have {self._num_outputs} outputs, got {targets.shape[1]}")
        return tuple(
            MetricValues.compute(y_true=targets[:, output], y_pred=predictions[:, output])
            for output in range(self._num_outputs)
        )

    def save(self, path: Path) -> None:
        """
        Write the fitted regressor to a file.

        The file holds the settings, the scaling statistics of the sequences and targets, the network weights and
        the training history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``SequenceRegressor.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: SummaryNetwork = self.network
        Checkpoint(
            task=TaskKind.REGRESSION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={
                "length": self._length,
                "channels": self._channels,
                "num_outputs": self._num_outputs,
                "is_target_vector": self._is_target_vector,
            },
            scalers={"input": self._scaler.to_state(), "target": self._target_scaler.to_state()},
            network=network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a regressor written by save, ready to predict without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted regressor.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a regressor checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.REGRESSION)
        settings: RegressorSettings = _CODEC.decode(RegressorSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        regressor: Self = cls(settings)
        regressor._length = checkpoint.read_int("length")
        regressor._channels = checkpoint.read_int("channels")
        regressor._num_outputs = checkpoint.read_int("num_outputs")
        regressor._is_target_vector = checkpoint.read_bool("is_target_vector")
        regressor._scaler.restore_state(checkpoint.read_scaler("input"), regressor._channels)
        regressor._target_scaler.restore_state(checkpoint.read_scaler("target"), regressor._num_outputs)
        network: SummaryNetwork = regressor._build_network()
        checkpoint.restore_network(network, regressor._trainer.device)
        regressor._network = network
        regressor._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return regressor

    def _build_network(self) -> SummaryNetwork:
        return self.settings.model.build_summary_network(
            SummaryShape(input_length=self._length, input_channels=self._channels, output_size=self._num_outputs)
        )

    @staticmethod
    def _validate_targets(y: FloatArray, num_sequences: int) -> FloatArray:
        targets: FloatArray = np.asarray(y)
        if not (np.issubdtype(targets.dtype, np.floating) or np.issubdtype(targets.dtype, np.integer)):
            raise TypeError(f"y must be numeric, got dtype {targets.dtype}")
        if targets.ndim not in (1, 2) or targets.shape[0] != num_sequences or 0 in targets.shape:
            raise ValueError(
                f"y must have shape ({num_sequences},) or ({num_sequences}, num_outputs), got {targets.shape}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("y contains NaN or infinite values.")
        return targets.reshape(num_sequences, -1).astype(np.float64)

    def _build_dataset(self, sequences: SequenceArray, targets: FloatArray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(self._scaler.transform(sequences.values)),
            torch.from_numpy(self._target_scaler.transform(targets)),
        )
