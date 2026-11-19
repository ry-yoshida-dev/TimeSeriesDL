from __future__ import annotations

from collections.abc import Sequence
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
from ...models import SequenceShape
from ...models.interfaces.sequence_generation_network import SequenceGenerationNetwork
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import Device
from ...training import Trainer, TrainingHistory
from .generation_step import GenerationStep
from .metrics import GenerationMetrics
from .settings import GeneratorSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(GeneratorSettings)


class SequenceGenerator:
    """
    Learns the distribution of fixed-length multivariate sequences and draws new synthetic ones.

    The sequences are scaled per channel with training statistics, the model is trained with its own objective, and
    samples are returned in the original scale. Training data are either sequences of equal length or windows cut
    from long series.

    Parameters
    ----------
    settings : GeneratorSettings
        The model, scaling, windows, evaluation and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels or is not invertible.
    """

    def __init__(self, settings: GeneratorSettings) -> None:
        self.settings: GeneratorSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(settings.scaling)
        if not self._scaler.is_invertible:
            raise ValueError(f"{type(settings.scaling).__name__} has no inverse; samples could not be unscaled.")
        self._trainer: Trainer = Trainer(settings.training)
        self._network: SequenceGenerationNetwork | None = None
        self._history: TrainingHistory | None = None
        self._length: int = 0
        self._channels: int = 0

    @property
    def network(self) -> SequenceGenerationNetwork:
        """
        Get the trained network.

        Returns
        -------
        SequenceGenerationNetwork
            The network that draws scaled sequences.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        if self._network is None:
            raise NotFittedError("SequenceGenerator is not fitted yet. Call fit() first.")
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
            raise NotFittedError("SequenceGenerator is not fitted yet. Call fit() first.")
        return self._history

    def fit(self, X: FloatArray, validation: FloatArray | None = None) -> TrainingHistory:
        """
        Train a new network on sequences.

        Parameters
        ----------
        X : FloatArray
            Training sequences with shape (num_sequences, length, channels).
        validation : FloatArray | None
            Sequences of the same length and channels whose loss selects the best epoch and stops training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        TypeError
            If the sequences are not numeric.
        ValueError
            If the sequences are invalid, the model does not support their shape, or ``training.max_epochs`` is
            below the epochs the model needs before it generates.
        """
        sequences: SequenceArray = SequenceArray(X)
        validation_sequences: SequenceArray | None = None
        if validation is not None:
            validation_sequences = SequenceArray(validation)
            validation_sequences.require_shape(sequences.length, sequences.channels)
        self._network = None
        self._history = None
        self._length = sequences.length
        self._channels = sequences.channels
        self._scaler.fit(sequences.values)
        self._trainer.seed()
        network: SequenceGenerationNetwork = self._build_network()
        if self.settings.training.max_epochs < network.minimum_epochs:
            raise ValueError(
                f"{type(self.settings.model).__name__} needs at least {network.minimum_epochs} epochs to generate, "
                + f"got max_epochs {self.settings.training.max_epochs}"
            )
        history: TrainingHistory = self._trainer.fit(
            network,
            GenerationStep(network=network),
            self._build_dataset(sequences),
            None if validation_sequences is None else self._build_dataset(validation_sequences),
        )
        self._network = network
        self._history = history
        return history

    def fit_series(
        self,
        series: FloatArray | Sequence[FloatArray],
        validation_series: FloatArray | Sequence[FloatArray] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network on the windows of long series, cut with ``window_length`` and ``window_stride``.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one window long.
        validation_series : FloatArray | Sequence[FloatArray] | None
            Series whose windows' loss selects the best epoch and stops training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If ``window_length`` is not set, a series is invalid or shorter than one window, or training is
            impossible as in ``fit``.
        """
        if self.settings.window_length is None:
            raise ValueError("window_length must be set in GeneratorSettings to fit on series.")
        windows: SlidingWindows = SlidingWindows(self.settings.window_length, self.settings.window_stride)
        return self.fit(
            self._cut_windows(SeriesCollection.from_input(series), windows),
            None
            if validation_series is None
            else self._cut_windows(SeriesCollection.from_input(validation_series), windows),
        )

    def generate(self, num_samples: int, random_seed: int | None = None) -> FloatArray:
        """
        Draw synthetic sequences.

        Parameters
        ----------
        num_samples : int
            Number of sequences.
        random_seed : int | None
            Seed of the samples, so the same seed draws the same sequences; PyTorch's global generator when None.

        Returns
        -------
        FloatArray
            The sequences in the original scale with shape (num_samples, length, channels).

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If num_samples is not positive.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        network: SequenceGenerationNetwork = self.network
        device: torch.device = self._trainer.device
        generator: torch.Generator | None = None
        if random_seed is not None:
            generator = torch.Generator(device=device).manual_seed(random_seed)
        batch_size: int = self.settings.training.batch_size
        batches: list[Tensor] = []
        _ = network.eval()
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                batches.append(network.sample(min(batch_size, num_samples - start), generator).cpu())
        return self._scaler.inverse_transform(torch.cat(batches).double().numpy())

    def evaluate(self, X: FloatArray, random_seed: int = 0) -> GenerationMetrics:
        """
        Compare as many synthetic sequences as given with real ones, in the scaled space.

        Parameters
        ----------
        X : FloatArray
            Real sequences with shape (num_sequences, length, channels), at least two, of at least two steps.
        random_seed : int
            Seed of the synthetic sequences.

        Returns
        -------
        GenerationMetrics
            The discriminative and predictive scores and the distribution distances.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the sequences are invalid, too few or too short, or have another length or number of channels.
        """
        _ = self.network
        real: SequenceArray = SequenceArray(X)
        real.require_shape(self._length, self._channels)
        synthetic: FloatArray = self.generate(real.num_sequences, random_seed)
        return GenerationMetrics.compute(
            self._scaler.transform(real.values), self._scaler.transform(synthetic), self.settings.evaluation
        )

    def save(self, path: Path) -> None:
        """
        Write the fitted generator to a file.

        The file holds the settings, the scaling statistics, the network weights and the training history, and loads
        without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``SequenceGenerator.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        network: SequenceGenerationNetwork = self.network
        Checkpoint(
            task=TaskKind.GENERATION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={"length": self._length, "channels": self._channels},
            scalers={"input": self._scaler.to_state()},
            network=network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a generator written by save, ready to generate without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted generator.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a generator checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.GENERATION)
        settings: GeneratorSettings = _CODEC.decode(GeneratorSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        generator: Self = cls(settings)
        generator._length = checkpoint.read_int("length")
        generator._channels = checkpoint.read_int("channels")
        generator._scaler.restore_state(checkpoint.read_scaler("input"), generator._channels)
        network: SequenceGenerationNetwork = generator._build_network()
        checkpoint.restore_network(network, generator._trainer.device)
        generator._network = network
        generator._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return generator

    def _build_network(self) -> SequenceGenerationNetwork:
        return self.settings.model.build_sequence_generation_network(
            SequenceShape(length=self._length, channels=self._channels)
        )

    def _build_dataset(self, sequences: SequenceArray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(self._scaler.transform(sequences.values)))

    @staticmethod
    def _cut_windows(collection: SeriesCollection, windows: SlidingWindows) -> FloatArray:
        if collection.shortest_length < windows.length:
            raise ValueError(
                f"every series must have at least {windows.length} steps, got {collection.shortest_length}"
            )
        return np.concatenate([windows.extract(values) for values in collection.series], axis=0)
