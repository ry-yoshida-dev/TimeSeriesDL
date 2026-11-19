from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from data_scaler import InvertibleScaler, Scaler, ScalerState, ScalingKind, ScalingParameters
from sklearn.exceptions import NotFittedError

from ..array_types import FloatArray


@dataclass(eq=False)
class ChannelScaler:
    """
    Scales each channel of time series with a ``data_scaler`` strategy fitted over all of their time steps.

    Values of any shape whose last dimension is the channel are accepted.

    Attributes
    ----------
    parameters : ScalingParameters
        The scaling strategy; it must transform each channel independently of the other channels of a time step.

    Raises
    ------
    ValueError
        If the strategy rescales each sample on its own.
    """

    parameters: ScalingParameters
    _scaler: Scaler = field(init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.parameters.kind == ScalingKind.SAMPLE_WISE:
            raise ValueError(
                f"{type(self.parameters).__name__} mixes the channels of a time step; choose a per-channel strategy."
            )
        self._scaler = self.parameters.build_scaler()

    @property
    def is_invertible(self) -> bool:
        """
        Check whether scaled values can be mapped back.

        Returns
        -------
        bool
            True if the strategy has an inverse transformation.
        """
        return isinstance(self._scaler, InvertibleScaler)

    def fit(self, values: FloatArray) -> None:
        """
        Learn the statistics of every channel.

        Parameters
        ----------
        values : FloatArray
            Values whose last dimension is the channel.
        """
        self._scaler.fit(self._flatten(values))
        self._is_fitted = True

    def transform(self, values: FloatArray) -> FloatArray:
        """
        Scale values with the fitted statistics.

        Parameters
        ----------
        values : FloatArray
            Values whose last dimension is the channel.

        Returns
        -------
        FloatArray
            The scaled values as float32, with the shape of the input.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        self._require_fitted()
        scaled: FloatArray = self._scaler.transform(self._flatten(values))
        return scaled.reshape(values.shape).astype(np.float32)

    def inverse_transform(self, values: FloatArray) -> FloatArray:
        """
        Map scaled values back to the original scale.

        Parameters
        ----------
        values : FloatArray
            Scaled values whose last dimension is the channel.

        Returns
        -------
        FloatArray
            The values in the original scale, with the shape of the input.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        TypeError
            If the strategy has no inverse.
        """
        self._require_fitted()
        if not isinstance(self._scaler, InvertibleScaler):
            raise TypeError(f"{type(self.parameters).__name__} has no inverse transformation.")
        restored: FloatArray = self._scaler.inverse_transform(self._flatten(values))
        return restored.reshape(values.shape)

    def to_state(self) -> ScalerState:
        """
        Export the fitted statistics.

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
        return self._scaler.to_state()

    def restore_state(self, state: ScalerState, channels: int) -> None:
        """
        Load fitted statistics in place of fitting.

        Parameters
        ----------
        state : ScalerState
            A state exported by ``to_state`` from a scaler with the same parameters.
        channels : int
            The number of channels the statistics must describe.

        Raises
        ------
        ValueError
            If the state belongs to other parameters or describes another number of channels.
        """
        if state.n_features != channels:
            raise ValueError(f"scaler state must describe {channels} channels, got {state.n_features}")
        self._scaler.restore_state(state)
        self._is_fitted = True

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("ChannelScaler is not fitted yet. Call fit() first.")

    @staticmethod
    def _flatten(values: FloatArray) -> FloatArray:
        return values.reshape(-1, values.shape[-1])
