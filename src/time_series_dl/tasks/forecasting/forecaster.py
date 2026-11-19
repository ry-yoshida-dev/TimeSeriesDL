from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch
from data_scaler import ScalerState
from sklearn.exceptions import NotFittedError
from torch import Tensor
from torch.utils.data import TensorDataset

from ...array_types import FloatArray
from ...data import (
    ChannelScaler,
    CovariateCollection,
    CovariateDimensions,
    CovariateMismatchError,
    CovariateScaler,
    SequenceArray,
    SeriesCollection,
    SeriesCovariates,
    SlidingWindows,
    WindowCovariates,
)
from ...models.interfaces import (
    CovariateForecastingModel,
    CovariateForecastNetwork,
    CovariateForecastShape,
    ForecastingModel,
    ForecastNetwork,
    ForecastShape,
)
from ...models.interfaces.protocols.generative_forecasting import GenerativeForecastingModel
from ...models.interfaces.protocols.published_weights import PublishedWeightsModel
from ...models.interfaces.protocols.zero_shot_forecasting import ZeroShotForecastingModel
from ...persistence import Checkpoint, EncodedValue, SettingsCodec, TaskKind
from ...runtime import BatchedInference, Device
from ...training import Trainer, TrainingHistory
from .backtest import ForecastBacktest
from .forecast_step import ForecastStep
from .instance_normalization import InstanceNormalization
from .metrics import ForecastMetrics
from .objectives import (
    GaussianObjective,
    LaplaceObjective,
    LowRankGaussianObjective,
    PointObjective,
    QuantileObjective,
    SampleObjective,
    StudentTObjective,
)
from .prediction import ForecastPrediction
from .settings import ForecasterSettings

_COVARIATE_SCALER_PREFIX: Final[str] = "covariate_"
_COVARIATE_VALUE: Final[str] = "covariates"

_CODEC: Final[SettingsCodec] = SettingsCodec.including(
    ForecasterSettings,
    GaussianObjective,
    LaplaceObjective,
    LowRankGaussianObjective,
    PointObjective,
    QuantileObjective,
    SampleObjective,
    StudentTObjective,
    CovariateDimensions,
)

_DEFAULT_EVALUATION_SAMPLES: Final[int] = 100


class Forecaster:
    """
    Forecasts the next ``horizon`` steps of multivariate series from their last ``input_length`` steps.

    Training windows slide over every training series at ``window.stride``. Series are scaled per channel with
    statistics of the training series, and forecasts are returned in the original scale.

    Covariates are optional: channels of the series that are not targets are past-observed covariates, and
    ``SeriesCovariates`` add known covariates (known for the past window and the horizon, such as calendar features or
    planned promotions) and static numeric and categorical features of each series. A forecaster fitted with
    covariates needs covariates of the same dimensions wherever it forecasts, and one fitted without them accepts
    none; otherwise ``CovariateMismatchError`` is raised. Models that implement ``CovariateForecastingModel`` (TiDE,
    Temporal Fusion Transformer, DeepAR) read covariates as in their papers; every other model reads them through
    ``CovariateAdapterNetwork``, except generative models, which read none.

    Every forecast can come with sample paths: generative models (TimeGrad, CSDI) draw them, and their point forecast
    and quantiles are statistics of the samples; other models draw them from the objective's distribution. Samples are
    drawn from a generator seeded with ``training.random_seed`` on every call, so forecasts are reproducible.

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
        self._covariate_scaler: CovariateScaler | None = None

    @property
    def network(self) -> ForecastNetwork | CovariateForecastNetwork:
        """
        Get the trained network.

        Returns
        -------
        ForecastNetwork | CovariateForecastNetwork
            The network; a covariate network when fitted with covariates.

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
        covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
        validation_covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
    ) -> TrainingHistory:
        """
        Train a new network on every window of the training series.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more training series with shape (length, channels).
        validation_series : FloatArray | Sequence[FloatArray] | None
            Series whose windows select the best epoch and stop training early.
        covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each training series, whose known covariates are as long as their series; their scaling and
            categories are learned here.
        validation_covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each validation series, required exactly when there are both covariates and validation
            series.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If a series is invalid or shorter than one window with its horizon, or a target channel is out of range.
        CovariateMismatchError
            If covariates are not aligned with their series, the validation covariates are missing, superfluous or of
            other dimensions, or they hold a category not seen in the training covariates.
        """
        training: SeriesCollection = SeriesCollection.from_input(series)
        validation: SeriesCollection | None = (
            None if validation_series is None else SeriesCollection.from_input(validation_series)
        )
        if validation is not None and validation.channels != training.channels:
            raise ValueError(f"validation series must have {training.channels} channels, got {validation.channels}")
        training_covariates: CovariateCollection | None = (
            None if covariates is None else CovariateCollection.from_input(covariates)
        )
        validation_covariate_collection: CovariateCollection | None = (
            None if validation_covariates is None else CovariateCollection.from_input(validation_covariates)
        )
        self._check_validation_covariates(validation, training_covariates, validation_covariate_collection)
        targets: tuple[int, ...] = self._resolve_targets(training.channels)
        self._step = None
        self._history = None
        stacked: FloatArray = training.concatenate()
        self._input_scaler.fit(stacked)
        self._target_scaler.fit(stacked[:, list(targets)])
        self._input_channels = training.channels
        self._covariate_scaler = None
        if training_covariates is not None:
            training_covariates.require_aligned(training)
            covariate_scaler: CovariateScaler = CovariateScaler(self.settings.covariate_scaling)
            covariate_scaler.fit(training_covariates)
            self._covariate_scaler = covariate_scaler
        training_data: TensorDataset = self._build_dataset(training, targets, training_covariates)
        validation_data: TensorDataset | None = (
            None if validation is None else self._build_dataset(validation, targets, validation_covariate_collection)
        )
        self._trainer.seed()
        step: ForecastStep = self._build_step(training.channels, targets)
        history: TrainingHistory = self._trainer.fit(step.network, step, training_data, validation_data)
        self._step = step
        self._history = history
        return history

    def prepare_zero_shot(self, series: FloatArray | Sequence[FloatArray]) -> None:
        """
        Make a pretrained model forecast without training.

        The scaling is fitted on the series and the network is built with the model's published weights; the
        history then has no epochs. ``fit`` afterwards fine-tunes from the published weights instead. The pretrained
        models scale each series themselves (Chronos by its mean absolute value, the others by its mean and standard
        deviation), so ``NoScalingParameters`` and ``is_instance_normalized=False`` feed them the values they were
        trained on; other settings change the forecasts of models whose own scaling is not affine invariant.
        Covariates are not supported, because the adapter that reads them must be trained.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels) to fit the scaling on, such as the series to forecast.

        Raises
        ------
        TypeError
            If the model does not forecast zero-shot.
        ValueError
            If the model has no published weights, the objective does not fit the model's outputs (a
            ``SampleObjective`` for generative models such as Chronos and Moirai; otherwise a ``PointObjective`` or a
            ``QuantileObjective`` of the model's ``zero_shot_quantile_levels``), a series is invalid, or a target
            channel is out of range.
        """
        model: ForecastingModel = self.settings.model
        if not isinstance(model, ZeroShotForecastingModel):
            raise TypeError(f"{type(model).__name__} does not forecast zero-shot; call fit() instead.")
        if not model.is_pretrained:
            raise ValueError(f"{type(model).__name__} names no published weights; set its weights or call fit().")
        if isinstance(model, GenerativeForecastingModel):
            if not isinstance(self.settings.objective, SampleObjective):
                raise ValueError(
                    f"{type(model).__name__} forecasts sample paths; zero-shot forecasts need SampleObjective, "
                    + f"got {type(self.settings.objective).__name__}"
                )
        elif not (
            isinstance(self.settings.objective, PointObjective)
            or (
                isinstance(self.settings.objective, QuantileObjective)
                and self.settings.objective.quantiles == model.zero_shot_quantile_levels
            )
        ):
            raise ValueError(
                "zero-shot forecasts need PointObjective or QuantileObjective with the levels "
                + f"{model.zero_shot_quantile_levels}, got {type(self.settings.objective).__name__}"
            )
        collection: SeriesCollection = SeriesCollection.from_input(series)
        targets: tuple[int, ...] = self._resolve_targets(collection.channels)
        self._step = None
        self._history = None
        stacked: FloatArray = collection.concatenate()
        self._input_scaler.fit(stacked)
        self._target_scaler.fit(stacked[:, list(targets)])
        self._input_channels = collection.channels
        self._covariate_scaler = None
        self._trainer.seed()
        step: ForecastStep = self._build_step(collection.channels, targets)
        _ = step.network.to(self._trainer.device).eval()
        self._step = step
        self._history = TrainingHistory(training_losses=(), validation_losses=(), best_epoch=0, is_stopped_early=False)

    def predict(
        self, past: FloatArray, covariates: WindowCovariates | None = None, num_samples: int = 0
    ) -> ForecastPrediction:
        """
        Forecast the horizon after each of a batch of past windows.

        Parameters
        ----------
        past : FloatArray
            Past windows with shape (num_windows, input_length, channels) in the original scale.
        covariates : WindowCovariates | None
            Covariates of each window, whose known covariates cover the window and its horizon
            (``input_length + horizon`` steps); required exactly when the forecaster was fitted with covariates.
        num_samples : int
            Number of sample paths per window to return in ``samples``; 0 for none. A generative model draws exactly
            these paths and derives the point forecast and the quantiles from them, or draws the objective's
            ``num_samples`` paths when 0.

        Returns
        -------
        ForecastPrediction
            The forecasts of each window.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the windows are invalid or have another length or number of channels, or num_samples is negative.
        CovariateMismatchError
            If the covariates are missing, superfluous, of other dimensions or lengths than in training, for another
            number of windows, or hold a category not seen in training.
        """
        step: ForecastStep = self._require_step()
        if num_samples < 0:
            raise ValueError(f"num_samples must not be negative, got {num_samples}")
        windows: SequenceArray = SequenceArray(past)
        windows.require_shape(self.settings.window.input_length, self._input_channels)
        inputs: tuple[Tensor, ...] = (
            torch.from_numpy(self._input_scaler.transform(windows.values)),
            *self._convert_covariates(covariates, windows.num_sequences),
        )
        generator: torch.Generator = self._create_generator()
        outputs: Tensor = BatchedInference(self._trainer.device, self.settings.training.batch_size).run(
            lambda batch: step.compute_outputs(batch, num_samples, generator), TensorDataset(*inputs)
        )
        unscaled: FloatArray = self._target_scaler.inverse_transform(outputs.permute(0, 1, 3, 2).numpy()).transpose(
            0, 1, 3, 2
        )
        levels: tuple[float, ...] = self.settings.objective.quantile_levels
        return ForecastPrediction(
            point=unscaled[..., 0],
            quantile_levels=levels,
            quantiles=unscaled[..., 1 : 1 + len(levels)],
            samples=unscaled[..., 1 + len(levels) :].transpose(0, 3, 1, 2) if num_samples > 0 else None,
        )

    def forecast(
        self, series: FloatArray, covariates: SeriesCovariates | None = None, num_samples: int = 0
    ) -> ForecastPrediction:
        """
        Forecast the horizon after the end of one series.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels); only its last ``input_length`` steps are read.
        covariates : SeriesCovariates | None
            Covariates of the series, whose known covariates are ``horizon`` steps longer than the series, so they
            cover the forecast horizon; required exactly when the forecaster was fitted with covariates.
        num_samples : int
            Number of sample paths to return, as in ``predict``.

        Returns
        -------
        ForecastPrediction
            The forecast, with one window.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If the series is invalid or shorter than ``input_length``, or num_samples is negative.
        CovariateMismatchError
            If the covariates are missing, superfluous or do not match the training covariates, or the known
            covariates do not extend ``horizon`` steps past the series.
        """
        collection: SeriesCollection = SeriesCollection.from_input(series)
        values: FloatArray = collection.series[0]
        length: int = self.settings.window.input_length
        if values.shape[0] < length:
            raise ValueError(f"series must have at least {length} steps, got {values.shape[0]}")
        window_covariates: WindowCovariates | None = None
        if covariates is not None:
            covariates.require_known_length(values.shape[0] + self.settings.window.horizon)
            window_covariates = covariates.slice_window(values.shape[0] - length, self.settings.window.total_length)
        return self.predict(values[np.newaxis, -length:], window_covariates, num_samples)

    def backtest(
        self,
        series: FloatArray | Sequence[FloatArray],
        covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
        num_samples: int = 0,
    ) -> ForecastBacktest:
        """
        Forecast every window of the series and pair the forecasts with their true continuations.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one window with its horizon long.
        covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each series, whose known covariates are as long as their series; required exactly when the
            forecaster was fitted with covariates.
        num_samples : int
            Number of sample paths per window, as in ``predict``.

        Returns
        -------
        ForecastBacktest
            The forecasts and the true future targets in the original scale, windows in the order of the series.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If a series is invalid or too short, or num_samples is negative.
        CovariateMismatchError
            If the covariates are missing, superfluous, misaligned or do not match the training covariates.
        """
        targets: tuple[int, ...] = self.target_channels
        collection: SeriesCollection = SeriesCollection.from_input(series)
        windows: FloatArray = self._extract_windows(collection)
        window_covariates: WindowCovariates | None = (
            None
            if covariates is None
            else CovariateCollection.from_input(covariates).extract_windows(self._sliding_windows(), collection)
        )
        length: int = self.settings.window.input_length
        prediction: ForecastPrediction = self.predict(windows[:, :length, :], window_covariates, num_samples)
        return ForecastBacktest(prediction=prediction, targets=windows[:, length:, :][..., list(targets)])

    def evaluate(
        self,
        series: FloatArray | Sequence[FloatArray],
        covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
        num_samples: int = _DEFAULT_EVALUATION_SAMPLES,
    ) -> ForecastMetrics:
        """
        Measure the forecasts of every window of the series against their true continuations.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels), each at least one window with its horizon long.
        covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each series, whose known covariates are as long as their series; required exactly when the
            forecaster was fitted with covariates.
        num_samples : int
            Number of sample paths per window for the sample CRPS; 0 to skip it (a generative model then draws the
            objective's number of samples for its point forecast and quantiles).

        Returns
        -------
        ForecastMetrics
            The metrics in the original scale.

        Raises
        ------
        NotFittedError
            If fit has not completed yet.
        ValueError
            If a series is invalid or too short, or num_samples is negative.
        CovariateMismatchError
            If the covariates are missing, superfluous, misaligned or do not match the training covariates.
        """
        result: ForecastBacktest = self.backtest(series, covariates, num_samples)
        return ForecastMetrics.compute(result.prediction, result.targets)

    def save(self, path: Path) -> None:
        """
        Write the fitted forecaster to a file.

        The file holds the settings, the scaling statistics of the series and the covariates, the categories of the
        static covariates, the network weights and the training history, and loads without pickle.

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
        values: dict[str, EncodedValue] = {"input_channels": self._input_channels}
        scalers: dict[str, ScalerState] = {
            "input": self._input_scaler.to_state(),
            "target": self._target_scaler.to_state(),
        }
        if self._covariate_scaler is not None:
            values[_COVARIATE_VALUE] = _CODEC.encode(self._covariate_scaler.dimensions)
            values["static_cardinalities"] = list(self._covariate_scaler.cardinalities)
            scalers.update(
                {_COVARIATE_SCALER_PREFIX + name: state for name, state in self._covariate_scaler.to_states().items()}
            )
        Checkpoint(
            task=TaskKind.FORECASTING,
            settings=_CODEC.encode(self.settings),
            history=_CODEC.encode(self.history),
            values=values,
            scalers=scalers,
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
        if checkpoint.contains(_COVARIATE_VALUE):
            covariate_scaler: CovariateScaler = CovariateScaler(settings.covariate_scaling)
            covariate_scaler.restore(
                _CODEC.decode(CovariateDimensions, checkpoint.values[_COVARIATE_VALUE]),
                checkpoint.read_int_tuple("static_cardinalities"),
                {
                    name.removeprefix(_COVARIATE_SCALER_PREFIX): state
                    for name, state in checkpoint.scalers.items()
                    if name.startswith(_COVARIATE_SCALER_PREFIX)
                },
            )
            forecaster._covariate_scaler = covariate_scaler
        step: ForecastStep = forecaster._build_step(input_channels, targets, is_restoring=True)
        checkpoint.restore_network(step.network, forecaster._trainer.device)
        forecaster._input_channels = input_channels
        forecaster._step = step
        forecaster._history = _CODEC.decode(TrainingHistory, checkpoint.history)
        return forecaster

    def _build_step(self, input_channels: int, targets: tuple[int, ...], is_restoring: bool = False) -> ForecastStep:
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
            network=self._build_network(shape, is_restoring),
            objective=self.settings.objective,
            target_channels=targets,
            normalization=InstanceNormalization() if self.settings.is_instance_normalized else None,
        )

    def _build_network(self, shape: ForecastShape, is_restoring: bool) -> ForecastNetwork | CovariateForecastNetwork:
        model: ForecastingModel = self.settings.model
        if is_restoring and isinstance(model, PublishedWeightsModel):
            # The checkpoint's weights replace the published ones, so they are not downloaded.
            model = model.without_pretrained_weights()
        if self._covariate_scaler is None:
            return model.build_forecast_network(shape)
        if isinstance(model, GenerativeForecastingModel):
            raise CovariateMismatchError(f"{type(model).__name__} is generative and reads no covariates; pass none.")
        dimensions: CovariateDimensions = self._covariate_scaler.dimensions
        # Normalized windows also pass the location and log scale of each target channel as static features.
        statistics: int = 2 * shape.output_channels if self.settings.is_instance_normalized else 0
        covariate_shape: CovariateForecastShape = CovariateForecastShape(
            forecast=shape,
            known_channels=dimensions.known_channels,
            static_channels=dimensions.static_channels + statistics,
            static_cardinalities=self._covariate_scaler.cardinalities,
        )
        if isinstance(model, CovariateForecastingModel):
            return model.build_covariate_forecast_network(covariate_shape)
        return self.settings.covariate_adapter.build_network(model, covariate_shape)

    def _resolve_targets(self, channels: int) -> tuple[int, ...]:
        targets: tuple[int, ...] = (
            tuple(range(channels)) if self.settings.target_channels is None else self.settings.target_channels
        )
        if any(channel < 0 or channel >= channels for channel in targets):
            raise ValueError(f"target_channels must be in [0, {channels}), got {targets}")
        return targets

    def _sliding_windows(self) -> SlidingWindows:
        return SlidingWindows(self.settings.window.total_length, self.settings.window.stride)

    def _extract_windows(self, collection: SeriesCollection) -> FloatArray:
        total: int = self.settings.window.total_length
        if collection.shortest_length < total:
            raise ValueError(f"every series must have at least {total} steps, got {collection.shortest_length}")
        windows: SlidingWindows = self._sliding_windows()
        return np.concatenate([windows.extract(values) for values in collection.series], axis=0)

    def _build_dataset(
        self, collection: SeriesCollection, targets: tuple[int, ...], covariates: CovariateCollection | None
    ) -> TensorDataset:
        windows: FloatArray = self._input_scaler.transform(self._extract_windows(collection))
        length: int = self.settings.window.input_length
        past: Tensor = torch.from_numpy(np.ascontiguousarray(windows[:, :length, :]))
        future: Tensor = torch.from_numpy(np.ascontiguousarray(windows[:, length:, :][..., list(targets)]))
        window_covariates: WindowCovariates | None = (
            None if covariates is None else covariates.extract_windows(self._sliding_windows(), collection)
        )
        return TensorDataset(past, *self._convert_covariates(window_covariates, past.shape[0]), future)

    def _convert_covariates(self, covariates: WindowCovariates | None, num_windows: int) -> tuple[Tensor, ...]:
        scaler: CovariateScaler | None = self._covariate_scaler
        if scaler is None:
            if covariates is not None:
                raise CovariateMismatchError("the forecaster was fitted without covariates; pass none.")
            return ()
        if covariates is None:
            raise CovariateMismatchError("the forecaster was fitted with covariates; pass them too.")
        if covariates.num_windows != num_windows:
            raise CovariateMismatchError(
                f"covariates of {num_windows} windows are required, got {covariates.num_windows}"
            )
        total: int = self.settings.window.total_length
        if covariates.known_length is not None and covariates.known_length != total:
            raise CovariateMismatchError(
                f"known covariates of each window must cover {total} steps, got {covariates.known_length}"
            )
        scaled: WindowCovariates = scaler.transform(covariates)
        return (
            self._to_tensor(scaled.known, (num_windows, total, 0)),
            self._to_tensor(scaled.static, (num_windows, 0)),
            torch.from_numpy(
                np.zeros((num_windows, 0), dtype=np.int64)
                if scaled.static_categories is None
                else np.ascontiguousarray(scaled.static_categories, dtype=np.int64)
            ),
        )

    @staticmethod
    def _to_tensor(values: FloatArray | None, empty_shape: tuple[int, ...]) -> Tensor:
        array: FloatArray = np.zeros(empty_shape, dtype=np.float32) if values is None else values
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    @staticmethod
    def _check_validation_covariates(
        validation: SeriesCollection | None,
        training_covariates: CovariateCollection | None,
        validation_covariates: CovariateCollection | None,
    ) -> None:
        if validation is None:
            if validation_covariates is not None:
                raise CovariateMismatchError("validation_covariates were given without validation_series.")
            return
        if (training_covariates is None) != (validation_covariates is None):
            raise CovariateMismatchError("validation_covariates must be given exactly when covariates are.")
        if (
            training_covariates is not None
            and validation_covariates is not None
            and training_covariates.dimensions != validation_covariates.dimensions
        ):
            raise CovariateMismatchError(
                f"validation covariates must have the dimensions {training_covariates.dimensions}, "
                + f"got {validation_covariates.dimensions}"
            )

    def _create_generator(self) -> torch.Generator:
        generator: torch.Generator = torch.Generator(device=self._trainer.device)
        seed: int | None = self.settings.training.random_seed
        _ = generator.seed() if seed is None else generator.manual_seed(seed)
        return generator

    def _require_step(self) -> ForecastStep:
        if self._step is None:
            raise NotFittedError("Forecaster is not fitted yet. Call fit() first.")
        return self._step
