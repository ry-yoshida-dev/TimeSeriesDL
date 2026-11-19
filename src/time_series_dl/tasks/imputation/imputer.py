from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ...array_types import BoolArray, FloatArray
from ...data import ChannelScaler, SequenceArray
from ...models import ReconstructionNetwork, SequenceShape
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .masked_reconstruction_step import MaskedReconstructionStep
from .metrics import ImputationMetrics
from .settings import ImputerSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(ImputerSettings)


class Imputer:
    """
    Fills missing values (NaN) of equal-length multivariate sequences.

    Parameters
    ----------
    settings : ImputerSettings
        The model, masking and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy is not invertible or mixes channels.
    """

    def __init__(self, settings: ImputerSettings) -> None:
        self.settings: ImputerSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(dataclasses.replace(settings.scaling, is_nan_allowed=True))
        if not self._scaler.is_invertible:
            raise ValueError(f"{type(settings.scaling).__name__} has no inverse; imputations could not be unscaled.")
        self._trainer: Trainer = Trainer(settings.training)
        self._step: MaskedReconstructionStep | None = None
        self._history: TrainingHistory | None = None
        self._length: int = 0
        self._channels: int = 0

    @property
    def network(self) -> ReconstructionNetwork:
        """
        Get the trained network.

        Returns
        -------
        ReconstructionNetwork
            The network.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        return self._require_step().network

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
            raise NotFittedError("Imputer is not fitted yet. Call fit() first.")
        return self._history

    def fit(self, X: FloatArray, validation: FloatArray | None = None) -> TrainingHistory:
        """
        Train a new network to reconstruct hidden observed values.

        Parameters
        ----------
        X : FloatArray
            Training sequences with shape (num_sequences, length, channels); NaN marks missing values.
        validation : FloatArray | None
            Sequences of the same shape whose fixed hidden values select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the sequences are not numeric.
        ValueError
            If the shapes are invalid, a value is infinite, or a channel has no observed value.
        """
        sequences: SequenceArray = SequenceArray(X, is_nan_allowed=True)
        validation_sequences: SequenceArray | None = None
        if validation is not None:
            validation_sequences = SequenceArray(validation, is_nan_allowed=True)
            validation_sequences.require_shape(sequences.length, sequences.channels)
        self._step = None
        self._history = None
        self._scaler.fit(sequences.values)
        self._length = sequences.length
        self._channels = sequences.channels
        self._trainer.seed()
        step: MaskedReconstructionStep = self._build_step()
        history: TrainingHistory = self._trainer.fit(
            step.network,
            step,
            self._build_dataset(sequences),
            None if validation_sequences is None else self._build_dataset(validation_sequences),
        )
        self._step = step
        self._history = history
        return history

    def impute(self, X: FloatArray) -> FloatArray:
        """
        Replace the missing values of sequences by their reconstruction.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels); NaN marks missing values.

        Returns
        -------
        FloatArray
            The sequences as float64, with observed values kept exactly and missing values imputed.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid or have another length or number of channels.
        """
        sequences: SequenceArray = SequenceArray(X, is_nan_allowed=True)
        reconstructed: FloatArray = self._reconstruct(sequences)
        original: FloatArray = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(original), reconstructed, original)

    def evaluate(self, X: FloatArray, is_hidden: BoolArray) -> ImputationMetrics:
        """
        Measure the imputation of known values hidden from the imputer.

        Parameters
        ----------
        X : FloatArray
            Sequences with shape (num_sequences, length, channels); NaN marks values that are missing anyway.
        is_hidden : BoolArray
            Mask of the shape of ``X`` marking the observed values to hide and evaluate.

        Returns
        -------
        ImputationMetrics
            The errors over the hidden observed values, in the original scale.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the shapes differ or no observed value is hidden.
        """
        sequences: SequenceArray = SequenceArray(X, is_nan_allowed=True)
        mask: BoolArray = np.asarray(is_hidden, dtype=np.bool_)
        if mask.shape != sequences.values.shape:
            raise ValueError(f"is_hidden must have shape {sequences.values.shape}, got {mask.shape}")
        is_evaluated: BoolArray = mask & ~np.isnan(sequences.values)
        if not np.any(is_evaluated):
            raise ValueError("is_hidden must mark at least one observed value.")
        hidden: FloatArray = np.where(mask, np.nan, sequences.values)
        reconstructed: FloatArray = self._reconstruct(SequenceArray(hidden, is_nan_allowed=True))
        errors: FloatArray = (reconstructed[is_evaluated] - sequences.values[is_evaluated]).astype(np.float64)
        mse: float = float(np.mean(np.square(errors)))
        return ImputationMetrics(
            num_values=int(errors.size), mae=float(np.mean(np.abs(errors))), mse=mse, rmse=float(np.sqrt(mse))
        )

    def save(self, path: Path) -> None:
        """
        Write the fitted imputer to a file.

        The file holds the settings, the scaling statistics, the network weights and the training history, and loads
        without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``Imputer.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        step: MaskedReconstructionStep = self._require_step()
        Checkpoint(
            task=TaskKind.IMPUTATION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={"length": self._length, "channels": self._channels},
            scalers={"input": self._scaler.to_state()},
            network=step.network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read an imputer written by save, ready to impute without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted imputer.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not an imputer checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.IMPUTATION)
        settings: ImputerSettings = _CODEC.decode(ImputerSettings, checkpoint.settings)
        if device is not None:
            settings = dataclasses.replace(settings, training=dataclasses.replace(settings.training, device=device))
        imputer: Self = cls(settings)
        imputer._length = checkpoint.read_int("length")
        imputer._channels = checkpoint.read_int("channels")
        imputer._scaler.restore_state(checkpoint.read_scaler("input"), imputer._channels)
        step: MaskedReconstructionStep = imputer._build_step()
        checkpoint.restore_network(step.network, imputer._trainer.device)
        imputer._step = step
        imputer._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return imputer

    def _build_step(self) -> MaskedReconstructionStep:
        return MaskedReconstructionStep(
            network=self.settings.model.build_reconstruction_network(
                SequenceShape(length=self._length, channels=self._channels)
            ),
            mask_ratio=self.settings.mask_ratio,
            loss=self.settings.loss,
        )

    def _reconstruct(self, sequences: SequenceArray) -> FloatArray:
        step: MaskedReconstructionStep = self._require_step()
        sequences.require_shape(self._length, self._channels)
        scaled: FloatArray = self._scaler.transform(sequences.values)
        is_observed: BoolArray = ~np.isnan(scaled)
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            step.reconstruct,
            TensorDataset(torch.from_numpy(np.nan_to_num(scaled, nan=0.0)), torch.from_numpy(is_observed)),
        )
        return self._scaler.inverse_transform(outputs.double().numpy())

    def _build_dataset(self, sequences: SequenceArray) -> TensorDataset:
        scaled: FloatArray = self._scaler.transform(sequences.values)
        is_observed: BoolArray = ~np.isnan(scaled)
        seed: int | None = self.settings.training.random_seed
        is_hidden: BoolArray = np.random.default_rng(seed).random(scaled.shape) < self.settings.mask_ratio
        return TensorDataset(
            torch.from_numpy(np.nan_to_num(scaled, nan=0.0)),
            torch.from_numpy(is_observed),
            torch.from_numpy(is_hidden),
        )

    def _require_step(self) -> MaskedReconstructionStep:
        if self._step is None:
            raise NotFittedError("Imputer is not fitted yet. Call fit() first.")
        return self._step
