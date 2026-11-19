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

from ...array_types import BoolArray, FloatArray, IndexArray
from ...data import ChannelScaler, SeriesCollection, SlidingWindows
from ...models import ForecastNetwork, ReconstructionNetwork, SequenceShape
from ...models.interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from ...persistence import Checkpoint, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .metrics import AnomalyDetectionMetrics
from .scoring_methods import AnomalyScoring, ForecastErrorScoring, NativeScoring, ReconstructionErrorScoring
from .scoring_step import AnomalyScoringStep
from .settings import AnomalyDetectorSettings

_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    AnomalyDetectorSettings, ForecastErrorScoring, NativeScoring, ReconstructionErrorScoring
)


class AnomalyDetector:
    """
    Scores each time step of a series by how far it departs from the normal behavior a network was trained on.

    The scoring method of the settings decides what a window's per-step scores are: the reconstruction loss, the
    forecast loss of the steps after a past window, or the model's own anomaly score. The score of a step is the mean
    of its scores over every window that scores it; steps that no window scores (the first past window under
    forecast-error scoring) receive the lowest score of the series. Steps whose score exceeds a quantile of the
    training scores are anomalous.

    Parameters
    ----------
    settings : AnomalyDetectorSettings
        The model, window, threshold and training settings.

    Raises
    ------
    ValueError
        If the scaling strategy mixes channels.
    """

    def __init__(self, settings: AnomalyDetectorSettings) -> None:
        self.settings: AnomalyDetectorSettings = settings
        self._scaler: ChannelScaler = ChannelScaler(settings.scaling)
        self._trainer: Trainer = Trainer(settings.training)
        self._step: AnomalyScoringStep | None = None
        self._history: TrainingHistory | None = None
        self._threshold: float = 0.0
        self._channels: int = 0

    @property
    def network(self) -> ReconstructionNetwork | ForecastNetwork | AnomalyScoringNetwork:
        """
        Get the trained network the model built.

        Returns
        -------
        ReconstructionNetwork | ForecastNetwork | AnomalyScoringNetwork
            The reconstruction or forecast network whose errors are the scores, or the model's own anomaly scoring
            network.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        return self._require_step().model_network

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
            raise NotFittedError("AnomalyDetector is not fitted yet. Call fit() first.")
        return self._history

    @property
    def threshold(self) -> float:
        """
        Get the score above which a time step is anomalous.

        Returns
        -------
        float
            The ``threshold_quantile`` quantile of the training scores.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        _ = self._require_step()
        return self._threshold

    def fit(
        self,
        series: FloatArray | Sequence[FloatArray],
        validation_series: FloatArray | Sequence[FloatArray] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network on series of normal behavior and set the threshold.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one sample long (``window_length`` steps,
            plus the horizon under forecast-error scoring).
        validation_series : FloatArray | Sequence[FloatArray] | None
            Normal series whose loss selects the best epoch and stops training early.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If a series is invalid or shorter than one window.
        """
        training: SeriesCollection = SeriesCollection.from_input(series)
        validation: SeriesCollection | None = (
            None if validation_series is None else SeriesCollection.from_input(validation_series)
        )
        if validation is not None and validation.channels != training.channels:
            raise ValueError(f"validation series must have {training.channels} channels, got {validation.channels}")
        self._step = None
        self._history = None
        self._scaler.fit(training.concatenate())
        self._channels = training.channels
        self._trainer.seed()
        step: AnomalyScoringStep = self._build_step()
        windows: SlidingWindows = SlidingWindows(self._sample_length, self.settings.training_stride)
        history: TrainingHistory = self._trainer.fit(
            step.network,
            step,
            self._build_dataset(training, windows),
            None if validation is None else self._build_dataset(validation, windows),
        )
        self._step = step
        self._history = history
        training_scores: FloatArray = np.concatenate([self.score(values) for values in training.series])
        self._threshold = float(np.quantile(training_scores, self.settings.threshold_quantile))
        return history

    def score(self, series: FloatArray) -> FloatArray:
        """
        Compute the anomaly score of every time step of a series.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels), at least one window long.

        Returns
        -------
        FloatArray
            The scores with shape (length,); higher is more anomalous.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the series is invalid, too short, or has another number of channels.
        """
        step: AnomalyScoringStep = self._require_step()
        values: FloatArray = self._validate_series(series)
        length: int = self._sample_length
        windows: SlidingWindows = SlidingWindows(length)
        # A series of exactly one window comes back as a read-only view, which torch cannot wrap.
        scaled: FloatArray = np.require(windows.extract(self._scaler.transform(values)), requirements="W")
        window_scores: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            step.compute_scores, TensorDataset(torch.from_numpy(scaled))
        )
        flat_scores: FloatArray = window_scores.double().numpy().ravel()
        is_scored: BoolArray = ~np.isnan(flat_scores)
        starts: IndexArray = windows.starts(values.shape[0])
        positions: IndexArray = (starts[:, np.newaxis] + np.arange(length)).ravel()
        totals: FloatArray = np.bincount(
            positions, weights=np.where(is_scored, flat_scores, 0.0), minlength=values.shape[0]
        )
        counts: FloatArray = np.bincount(positions, weights=is_scored.astype(np.float64), minlength=values.shape[0])
        is_step_scored: BoolArray = counts > 0.0
        scores: FloatArray = np.divide(totals, counts, out=np.zeros_like(totals), where=is_step_scored)
        scores[~is_step_scored] = scores[is_step_scored].min()
        return scores

    def detect(self, series: FloatArray) -> BoolArray:
        """
        Detect the anomalous time steps of a series.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels), at least one window long.

        Returns
        -------
        BoolArray
            True for each step whose score exceeds the threshold, with shape (length,).

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the series is invalid, too short, or has another number of channels.
        """
        return self.score(series) > self.threshold

    def evaluate(self, series: FloatArray, is_anomalous: BoolArray) -> AnomalyDetectionMetrics:
        """
        Measure the detections of a series against labeled anomalous time steps.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels), at least one window long.
        is_anomalous : BoolArray
            True labels with shape (length,).

        Returns
        -------
        AnomalyDetectionMetrics
            The metrics.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the series is invalid or the labels do not match it.
        """
        scores: FloatArray = self.score(series)
        return AnomalyDetectionMetrics.compute(
            np.asarray(is_anomalous, dtype=np.bool_), scores > self.threshold, scores
        )

    def save(self, path: Path) -> None:
        """
        Write the fitted detector to a file.

        The file holds the settings, the threshold, the scaling statistics, the network weights and the training
        history, and loads without pickle.

        Parameters
        ----------
        path : Path
            The file to write; ``AnomalyDetector.load`` reads it back.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        """
        step: AnomalyScoringStep = self._require_step()
        Checkpoint(
            task=TaskKind.ANOMALY_DETECTION,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values={"channels": self._channels, "threshold": self._threshold},
            scalers={"input": self._scaler.to_state()},
            network=step.network.state_dict(),
        ).save(path)

    @classmethod
    def load(cls, path: Path, device: Device | None = None) -> Self:
        """
        Read a detector written by save, ready to score without fitting.

        Parameters
        ----------
        path : Path
            The file to read.
        device : Device | None
            The device to run on in place of the saved ``training.device``; the saved one when None.

        Returns
        -------
        Self
            The fitted detector.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not an anomaly detector checkpoint or its contents are inconsistent.
        """
        checkpoint: Checkpoint = Checkpoint.load(path, TaskKind.ANOMALY_DETECTION)
        settings: AnomalyDetectorSettings = _CODEC.decode(AnomalyDetectorSettings, checkpoint.settings)
        if device is not None:
            settings = replace(settings, training=replace(settings.training, device=device))
        detector: Self = cls(settings)
        detector._channels = checkpoint.read_int("channels")
        detector._threshold = checkpoint.read_float("threshold")
        detector._scaler.restore_state(checkpoint.read_scaler("input"), detector._channels)
        step: AnomalyScoringStep = detector._build_step()
        checkpoint.restore_network(step.network, detector._trainer.device)
        detector._step = step
        detector._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return detector

    @property
    def _scoring(self) -> AnomalyScoring:
        return self.settings.resolved_scoring

    @property
    def _sample_length(self) -> int:
        return self._scoring.compute_sample_length(self.settings.window_length)

    def _build_step(self) -> AnomalyScoringStep:
        return self._scoring.build_step(
            self.settings.model,
            SequenceShape(length=self.settings.window_length, channels=self._channels),
            self.settings.loss,
        )

    def _validate_series(self, series: FloatArray) -> FloatArray:
        values: FloatArray = SeriesCollection.from_input(series).series[0]
        if values.shape[1] != self._channels:
            raise ValueError(f"series must have {self._channels} channels, got {values.shape[1]}")
        if values.shape[0] < self._sample_length:
            raise ValueError(f"series must have at least {self._sample_length} steps, got {values.shape[0]}")
        return values

    def _build_dataset(self, collection: SeriesCollection, windows: SlidingWindows) -> TensorDataset:
        if collection.shortest_length < windows.length:
            raise ValueError(
                f"every series must have at least {windows.length} steps, got {collection.shortest_length}"
            )
        scaled: list[FloatArray] = [windows.extract(self._scaler.transform(values)) for values in collection.series]
        return TensorDataset(torch.from_numpy(np.concatenate(scaled, axis=0)))

    def _require_step(self) -> AnomalyScoringStep:
        if self._step is None:
            raise NotFittedError("AnomalyDetector is not fitted yet. Call fit() first.")
        return self._step
