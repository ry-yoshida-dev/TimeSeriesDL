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

from ....array_types import FloatArray
from ....data import ChannelScaler
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
from ....persistence import Checkpoint, SettingsCodec, TaskKind
from ....runtime import BatchedInference, Device
from ....training import Trainer, TrainingHistory
from .regression_step import IrregularRegressionStep
from .settings import IrregularRegressorSettings

# The irregular models are listed until ModelArchitecture names them, which adds them to every codec.
_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    IrregularRegressorSettings, MTANParameters, NeuralCDEParameters, ODERNNParameters, LatentODEParameters
)


class IrregularSequenceRegressor:
    """
    Predicts one or more values from each irregularly sampled multivariate sequence, such as the length of stay from
    ICU records.

    Values are scaled per channel with statistics of the observed training values, times so that the training times
    span [0, 1], and targets per output; predictions are returned in the original scale. Metrics are those of
    ``classreg.regression``, one per output.

    Parameters
    ----------
    settings : IrregularRegressorSettings
        The model, loss and training settings.

    Raises
    ------
    ValueError
        If a scaling strategy mixes channels, or the target scaling is not invertible.
    """

    def __init__(self, settings: IrregularRegressorSettings) -> None:
        self.settings: IrregularRegressorSettings = settings
        self._scaler: IrregularScaler = IrregularScaler(settings.scaling)
        self._target_scaler: ChannelScaler = ChannelScaler(settings.target_scaling)
        if not self._target_scaler.is_invertible:
            raise ValueError(
                f"{type(settings.target_scaling).__name__} has no inverse; predictions could not be unscaled."
            )
        self._trainer: Trainer = Trainer(settings.training)
        self._network: IrregularSummaryNetwork | None = None
        self._history: TrainingHistory | None = None
        self._is_target_vector: bool = False
        self._num_outputs: int = 0
        self._channels: int = 0

    @property
    def network(self) -> IrregularSummaryNetwork:
        """
        Get the trained network.

        Returns
        -------
        IrregularSummaryNetwork
            The network producing the scaled targets.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("IrregularSequenceRegressor is not fitted yet. Call fit() first.")
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
            raise NotFittedError("IrregularSequenceRegressor is not fitted yet. Call fit() first.")
        return self._history

    def fit(
        self,
        X: IrregularSequenceCollection,
        y: FloatArray,
        validation: tuple[IrregularSequenceCollection, FloatArray] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Training sequences.
        y : FloatArray
            Targets with shape (num_sequences,) or (num_sequences, num_outputs).
        validation : tuple[IrregularSequenceCollection, FloatArray] | None
            Sequences and targets that select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the targets are not numeric.
        ValueError
            If the shapes are invalid, a target is not finite or a channel is never observed.
        """
        targets: FloatArray = self._validate_targets(y, X.num_sequences)
        validation_data: tuple[IrregularSequenceCollection, FloatArray] | None = None
        if validation is not None:
            validation[0].require_channels(X.channels)
            validation_targets: FloatArray = self._validate_targets(validation[1], validation[0].num_sequences)
            if validation_targets.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"validation targets must have {targets.shape[1]} outputs, got {validation_targets.shape[1]}"
                )
            validation_data = (validation[0], validation_targets)
        self._network = None
        self._history = None
        self._is_target_vector = np.asarray(y).ndim == 2
        self._num_outputs = int(targets.shape[1])
        self._channels = X.channels
        self._scaler.fit(X.pad())
        self._target_scaler.fit(targets)
        self._trainer.seed()
        network: IrregularSummaryNetwork = self._build_network()
        history: TrainingHistory = self._trainer.fit(
            network,
            IrregularRegressionStep(network=network, loss=self.settings.loss),
            self._build_dataset(X, targets),
            None if validation_data is None else self._build_dataset(*validation_data),
        )
        self._network = network
        self._history = history
        return history

    def predict(self, X: IrregularSequenceCollection) -> FloatArray:
        """
        Predict the targets of each sequence.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Sequences.

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
            If the sequences have another number of channels.
        """
        network: IrregularSummaryNetwork = self.network
        X.require_channels(self._channels)
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: network(IrregularTensors.from_batch(batch).trimmed()),
            TensorDataset(*self._scaler.transform(X.pad()).to_tensors()),
        )
        predictions: FloatArray = self._target_scaler.inverse_transform(outputs.double().numpy())
        return predictions if self._is_target_vector else predictions[:, 0]

    def evaluate(self, X: IrregularSequenceCollection, y: FloatArray) -> tuple[MetricValues, ...]:
        """
        Measure the fitted regressor on sequences with known targets.

        Parameters
        ----------
        X : IrregularSequenceCollection
            Sequences.
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

        The file holds the settings, the scaling statistics of the values and targets, the time range, the network
        weights and the training history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``IrregularSequenceRegressor.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: IrregularSummaryNetwork = self.network
        Checkpoint(
            task=TaskKind.IRREGULAR_REGRESSION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={
                "channels": self._channels,
                "num_outputs": self._num_outputs,
                "is_target_vector": self._is_target_vector,
                "time_offset": self._scaler.time_offset,
                "time_span": self._scaler.time_span,
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
            If the file is not an irregular regressor checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.IRREGULAR_REGRESSION)
        settings: IrregularRegressorSettings = _CODEC.decode(IrregularRegressorSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        regressor: Self = cls(settings)
        regressor._channels = checkpoint.read_int("channels")
        regressor._num_outputs = checkpoint.read_int("num_outputs")
        regressor._is_target_vector = checkpoint.read_bool("is_target_vector")
        regressor._scaler.restore_state(
            checkpoint.read_scaler("input"),
            regressor._channels,
            checkpoint.read_float("time_offset"),
            checkpoint.read_float("time_span"),
        )
        regressor._target_scaler.restore_state(checkpoint.read_scaler("target"), regressor._num_outputs)
        network: IrregularSummaryNetwork = regressor._build_network()
        checkpoint.restore_network(network, regressor._trainer.device)
        regressor._network = network
        regressor._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return regressor

    def _build_network(self) -> IrregularSummaryNetwork:
        return self.settings.model.build_irregular_summary_network(
            IrregularSummaryShape(input_channels=self._channels, output_size=self._num_outputs)
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

    def _build_dataset(self, sequences: IrregularSequenceCollection, targets: FloatArray) -> TensorDataset:
        return TensorDataset(
            *self._scaler.transform(sequences.pad()).to_tensors(),
            torch.from_numpy(self._target_scaler.transform(targets)),
        )
