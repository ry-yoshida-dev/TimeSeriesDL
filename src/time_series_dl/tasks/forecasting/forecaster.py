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
from ...models import ForecastNetwork, ForecastShape
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .forecast_step import ForecastStep
from .instance_normalization import InstanceNormalization
from .metrics import ForecastMetrics
from .objectives import GaussianObjective, LaplaceObjective, PointObjective, QuantileObjective, StudentTObjective
from .prediction import ForecastPrediction
from .settings import ForecasterSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    ForecasterSettings, GaussianObjective, LaplaceObjective, PointObjective, QuantileObjective, StudentTObjective
)


class Forecaster:
    """
    Forecasts the next ``horizon`` steps of multivariate series from their last ``input_length`` steps.

    Training windows slide over every training series at ``window.stride``. Series are scaled per channel with
    statistics of the training series, and forecasts are returned in the original scale.

    Parameters
    ----------
    settings : ForecasterSettings
        The model, window, objective and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy is not invertible or mixes channels.
    """

    def __init__(self, settings: ForecasterSettings) -> None:
        self.settings: ForecasterSettings = settings
        self._input_scaler: ChannelScaler = ChannelScaler(settings.scaling)
        self._target_scaler: ChannelScaler = ChannelScaler(settings.scaling)
        if not self._input_scaler.is_invertible:
            raise ValueError(f"{type(settings.scaling).__name__} has no inverse; forecasts could not be unscaled.")
        self._trainer: Trainer = Trainer(settings.training)
        self._step: ForecastStep | None = None
        self._history: TrainingHistory | None = None
        self._input_channels: int = 0

    @property
    def network(self) -> ForecastNetwork:
        """
        Get the trained network.

        Returns
        -------
        ForecastNetwork
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
            raise NotFittedError("Forecaster is not fitted yet. Call fit() first.")
        return self._history

    @property
    def target_channels(self) -> tuple[int, ...]:
        """
        Get the indices of the forecast channels.

        Returns
        -------
        tuple[int, ...]
            The indices, in the order of the forecast channels.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        return self._require_step().target_channels

    def fit(
        self,
        series: FloatArray | Sequence[FloatArray],
        validation_series: FloatArray | Sequence[FloatArray] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network on every window of the training series.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more training series with shape (length, channels).
        validation_series : FloatArray | Sequence[FloatArray] | None
            Series whose windows select the best epoch and stop training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If a series is invalid or shorter than one window with its horizon, or a target channel is out of range.
        """
        training: SeriesCollection = SeriesCollection.from_input(series)
        validation: SeriesCollection | None = (
            None if validation_series is None else SeriesCollection.from_input(validation_series)
        )
        if validation is not None and validation.channels != training.channels:
            raise ValueError(f"validation series must have {training.channels} channels, got {validation.channels}")
        targets: tuple[int, ...] = self._resolve_targets(training.channels)
        self._step = None
        self._history = None
        stacked: FloatArray = training.concatenate()
        self._input_scaler.fit(stacked)
        self._target_scaler.fit(stacked[:, list(targets)])
        self._input_channels = training.channels
        self._trainer.seed()
        step: ForecastStep = self._build_step(training.channels, targets)
        history: TrainingHistory = self._trainer.fit(
            step.network,
            step,
            self._build_dataset(training, targets),
            None if validation is None else self._build_dataset(validation, targets),
        )
        self._step = step
        self._history = history
        return history

    def predict(self, past: FloatArray) -> ForecastPrediction:
        """
        Forecast the horizon after each of a batch of past windows.

        Parameters
        ----------
        past : FloatArray
            Past windows with shape (num_windows, input_length, channels) in the original scale.

        Returns
        -------
        ForecastPrediction
            The forecasts of each window.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the windows are invalid or have another length or number of channels.
        """
        step: ForecastStep = self._require_step()
        windows: SequenceArray = SequenceArray(past)
        windows.require_shape(self.settings.window.input_length, self._input_channels)
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            step.compute_outputs, TensorDataset(torch.from_numpy(self._input_scaler.transform(windows.values)))
        )
        unscaled: FloatArray = self._target_scaler.inverse_transform(outputs.permute(0, 1, 3, 2).numpy()).transpose(
            0, 1, 3, 2
        )
        return ForecastPrediction(
            point=unscaled[..., 0],
            quantile_levels=self.settings.objective.quantile_levels,
            quantiles=unscaled[..., 1:],
        )

    def forecast(self, series: FloatArray) -> ForecastPrediction:
        """
        Forecast the horizon after the end of one series.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels); only its last ``input_length`` steps are read.

        Returns
        -------
        ForecastPrediction
            The forecast, with one window.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the series is invalid or shorter than ``input_length``.
        """
        collection: SeriesCollection = SeriesCollection.from_input(series)
        values: FloatArray = collection.series[0]
        length: int = self.settings.window.input_length
        if values.shape[0] < length:
            raise ValueError(f"series must have at least {length} steps, got {values.shape[0]}")
        return self.predict(values[np.newaxis, -length:])

    def evaluate(self, series: FloatArray | Sequence[FloatArray]) -> ForecastMetrics:
        """
        Measure the forecasts of every window of the series against their true continuations.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one window with its horizon long.

        Returns
        -------
        ForecastMetrics
            The metrics in the original scale.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If a series is invalid or too short.
        """
        targets: tuple[int, ...] = self.target_channels
        collection: SeriesCollection = SeriesCollection.from_input(series)
        windows: FloatArray = self._extract_windows(collection)
        length: int = self.settings.window.input_length
        prediction: ForecastPrediction = self.predict(windows[:, :length, :])
        return ForecastMetrics.compute(prediction, windows[:, length:, :][..., list(targets)])

    def save(self, path: Path) -> None:
        """
        Write the fitted forecaster to a file.

        The file holds the settings, the scaling statistics, the network weights and the training history, and loads
        without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``Forecaster.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        step: ForecastStep = self._require_step()
        Checkpoint(
            task=TaskKind.FORECASTING,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={"input_channels": self._input_channels},
            scalers={"input": self._input_scaler.to_state(), "target": self._target_scaler.to_state()},
            network=step.network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a forecaster written by save, ready to predict without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted forecaster.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a forecaster checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.FORECASTING)
        settings: ForecasterSettings = _CODEC.decode(ForecasterSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        forecaster: Self = cls(settings)
        input_channels: int = checkpoint.read_int("input_channels")
        targets: tuple[int, ...] = forecaster._resolve_targets(input_channels)
        forecaster._input_scaler.restore_state(checkpoint.read_scaler("input"), input_channels)
        forecaster._target_scaler.restore_state(checkpoint.read_scaler("target"), len(targets))
        step: ForecastStep = forecaster._build_step(input_channels, targets)
        checkpoint.restore_network(step.network, forecaster._trainer.device)
        forecaster._input_channels = input_channels
        forecaster._step = step
        forecaster._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return forecaster

    def _build_step(self, input_channels: int, targets: tuple[int, ...]) -> ForecastStep:
        shape: ForecastShape = ForecastShape(
            input_length=self.settings.window.input_length,
            input_channels=input_channels,
            horizon=self.settings.window.horizon,
            target_channels=targets,
            num_parameters=self.settings.objective.num_parameters,
            location_index=self.settings.objective.location_index,
            level_parameters=self.settings.objective.level_parameters,
        )
        return ForecastStep(
            network=self.settings.model.build_forecast_network(shape),
            objective=self.settings.objective,
            target_channels=targets,
            normalization=InstanceNormalization() if self.settings.is_instance_normalized else None,
        )

    def _resolve_targets(self, channels: int) -> tuple[int, ...]:
        targets: tuple[int, ...] = (
            tuple(range(channels)) if self.settings.target_channels is None else self.settings.target_channels
        )
        if any(channel < 0 or channel >= channels for channel in targets):
            raise ValueError(f"target_channels must be in [0, {channels}), got {targets}")
        return targets

    def _extract_windows(self, collection: SeriesCollection) -> FloatArray:
        total: int = self.settings.window.total_length
        if collection.shortest_length < total:
            raise ValueError(f"every series must have at least {total} steps, got {collection.shortest_length}")
        windows: SlidingWindows = SlidingWindows(total, self.settings.window.stride)
        return np.concatenate([windows.extract(values) for values in collection.series], axis=0)

    def _build_dataset(self, collection: SeriesCollection, targets: tuple[int, ...]) -> TensorDataset:
        windows: FloatArray = self._input_scaler.transform(self._extract_windows(collection))
        length: int = self.settings.window.input_length
        past: Tensor = torch.from_numpy(np.ascontiguousarray(windows[:, :length, :]))
        future: Tensor = torch.from_numpy(np.ascontiguousarray(windows[:, length:, :][..., list(targets)]))
        return TensorDataset(past, future)

    def _require_step(self) -> ForecastStep:
        if self._step is None:
            raise NotFittedError("Forecaster is not fitted yet. Call fit() first.")
        return self._step
