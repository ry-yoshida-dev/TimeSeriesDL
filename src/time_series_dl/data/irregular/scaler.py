from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from data_scaler import ScalerState, ScalingParameters
from sklearn.exceptions import NotFittedError

from ...array_types import FloatArray
from ..channel_scaler import ChannelScaler
from .padded_sequences import PaddedIrregularSequences


@dataclass(eq=False)
class IrregularScaler:
    """
    Scales irregularly sampled sequences with statistics of the training sequences: each channel's values with a
    ``data_scaler`` strategy fitted on its observed values only, and the times linearly so that the training times
    span [0, 1].

    Attributes
    ----------
    parameters : ScalingParameters
        The scaling strategy of the values; it must transform each channel independently, and NaN marks unobserved
        values whatever its ``is_nan_allowed``.

    Raises
    ------
    ValueError
        If the strategy rescales each sample on its own.
    """

    parameters: ScalingParameters
    _values: ChannelScaler = field(init=False, repr=False)
    _time_offset: float = field(default=0.0, init=False, repr=False)
    _time_span: float = field(default=1.0, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._values = ChannelScaler(replace(self.parameters, is_nan_allowed=True))

    @property
    def is_invertible(self) -> bool:
        """
        Check whether scaled values can be mapped back.

        Returns
        -------
        bool
            True if the strategy has an inverse transformation.
        """
        return self._values.is_invertible

    @property
    def time_offset(self) -> float:
        """
        Get the time that maps to 0, the earliest training time.

        Returns
        -------
        float
            The offset.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        return self._time_offset

    @property
    def time_span(self) -> float:
        """
        Get the time difference that maps to 1, the span of the training times (1 when they are all equal).

        Returns
        -------
        float
            The span.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        return self._time_span

    def fit(self, sequences: PaddedIrregularSequences) -> None:
        """
        Learn the statistics of the values and the range of the times.

        Parameters
        ----------
        sequences : PaddedIrregularSequences
            The training sequences.

        Raises
        ------
        ValueError
            If a channel is never observed.
        """
        self._values.fit(sequences.values)
        real_times: FloatArray = sequences.times[sequences.step_mask]
        start: float = float(real_times.min())
        span: float = float(real_times.max()) - start
        self._time_offset = start
        self._time_span = span if span > 0.0 else 1.0
        self._is_fitted = True

    def transform(self, sequences: PaddedIrregularSequences) -> PaddedIrregularSequences:
        """
        Scale the times and values.

        Parameters
        ----------
        sequences : PaddedIrregularSequences
            The sequences.

        Returns
        -------
        PaddedIrregularSequences
            The scaled sequences; unobserved values stay NaN.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        return sequences.with_arrays(self.transform_times(sequences.times), self._values.transform(sequences.values))

    def transform_times(self, times: FloatArray) -> FloatArray:
        """
        Scale times.

        Parameters
        ----------
        times : FloatArray
            Times of any shape.

        Returns
        -------
        FloatArray
            The scaled times.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        return (times - self._time_offset) / self._time_span

    def inverse_transform_values(self, values: FloatArray) -> FloatArray:
        """
        Map scaled values back to the original scale.

        Parameters
        ----------
        values : FloatArray
            Scaled values whose last dimension is the channel.

        Returns
        -------
        FloatArray
            The values in the original scale.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        TypeError
            If the strategy has no inverse.
        """
        return self._values.inverse_transform(values)

    def to_state(self) -> ScalerState:
        """
        Export the fitted statistics of the values; the time range is read from ``time_offset`` and ``time_span``.

        Returns
        -------
        ScalerState
            The state, which ``restore_state`` loads back.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        return self._values.to_state()

    def restore_state(self, state: ScalerState, channels: int, time_offset: float, time_span: float) -> None:
        """
        Load fitted statistics in place of fitting.

        Parameters
        ----------
        state : ScalerState
            A state exported by ``to_state`` from a scaler with the same parameters.
        channels : int
            The number of channels the statistics must describe.
        time_offset : float
            The saved ``time_offset``.
        time_span : float
            The saved ``time_span``.

        Raises
        ------
        ValueError
            If the state describes another number of channels, or the time range is not finite and positive.
        """
        if not (np.isfinite(time_offset) and np.isfinite(time_span) and time_span > 0.0):
            raise ValueError(f"time range must be finite with a positive span, got {time_offset} and {time_span}")
        self._values.restore_state(state, channels)
        self._time_offset = time_offset
        self._time_span = time_span
        self._is_fitted = True

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("IrregularScaler is not fitted yet. Call fit() first.")
