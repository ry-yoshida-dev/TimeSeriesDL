from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from data_scaler import ScalerState, ScalingParameters
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from ...array_types import FloatArray
from ..channel_scaler import ChannelScaler
from .collection import CovariateCollection
from .dimensions import CovariateDimensions
from .mismatch_error import CovariateMismatchError
from .window_covariates import WindowCovariates

KNOWN_STATE: Final[str] = "known"
STATIC_STATE: Final[str] = "static"


@dataclass(eq=False)
class CovariateScaler:
    """
    Scales covariates with statistics of the training covariates, and checks that later covariates have the same
    dimensions and only categories seen in training.

    Known covariates are scaled per channel over every time step of every training series, and numeric static features
    per feature over the training series. Categorical codes are not scaled; the number of categories of each feature is
    one more than its largest training code.

    Attributes
    ----------
    parameters : ScalingParameters
        The scaling strategy of the known covariates and of the numeric static features; it must transform each
        channel on its own.

    Raises
    ------
    ValueError
        If the strategy rescales each sample on its own.
    """

    parameters: ScalingParameters
    _known_scaler: ChannelScaler = field(init=False, repr=False)
    _static_scaler: ChannelScaler = field(init=False, repr=False)
    _dimensions: CovariateDimensions | None = field(default=None, init=False, repr=False)
    _cardinalities: tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._known_scaler = ChannelScaler(self.parameters)
        self._static_scaler = ChannelScaler(self.parameters)

    @property
    def dimensions(self) -> CovariateDimensions:
        """
        Get the dimensions of the training covariates.

        Returns
        -------
        CovariateDimensions
            The dimensions.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        if self._dimensions is None:
            raise NotFittedError("CovariateScaler is not fitted yet. Call fit() first.")
        return self._dimensions

    @property
    def cardinalities(self) -> tuple[int, ...]:
        """
        Get the number of categories of each categorical static feature.

        Returns
        -------
        tuple[int, ...]
            One more than the largest training code of each feature.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        _ = self.dimensions
        return self._cardinalities

    def fit(self, covariates: CovariateCollection) -> None:
        """
        Learn the statistics and categories of the training covariates.

        Parameters
        ----------
        covariates : CovariateCollection
            The covariates of every training series.
        """
        known: FloatArray | None = covariates.concatenate_known()
        static: FloatArray | None = covariates.stack_static()
        codes: NDArray[np.integer] | None = covariates.stack_static_categories()
        if known is not None:
            self._known_scaler.fit(known)
        if static is not None:
            self._static_scaler.fit(static)
        self._cardinalities = () if codes is None else tuple(int(code) + 1 for code in codes.max(axis=0))
        self._dimensions = covariates.dimensions

    def transform(self, covariates: WindowCovariates) -> WindowCovariates:
        """
        Scale the covariates of a batch of windows.

        Parameters
        ----------
        covariates : WindowCovariates
            Covariates in the original scale.

        Returns
        -------
        WindowCovariates
            The scaled covariates, with the categorical codes unchanged.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        CovariateMismatchError
            If the dimensions differ from the training covariates or a category was not seen in training.
        """
        if covariates.dimensions != self.dimensions:
            raise CovariateMismatchError(
                f"covariates must have the dimensions of the training covariates {self.dimensions}, "
                + f"got {covariates.dimensions}"
            )
        codes: NDArray[np.integer] | None = covariates.static_categories
        if codes is not None:
            is_unknown: NDArray[np.bool_] = codes >= np.asarray(self._cardinalities)
            if np.any(is_unknown):
                features: list[int] = np.flatnonzero(is_unknown.any(axis=0)).tolist()
                raise CovariateMismatchError(
                    f"static_categories {features} contain codes not seen in training; "
                    + f"the numbers of categories are {self._cardinalities}"
                )
        return WindowCovariates(
            known=None if covariates.known is None else self._known_scaler.transform(covariates.known),
            static=None if covariates.static is None else self._static_scaler.transform(covariates.static),
            static_categories=codes,
        )

    def to_states(self) -> dict[str, ScalerState]:
        """
        Export the fitted statistics.

        Returns
        -------
        dict[str, ScalerState]
            The states of the scalers in use, which ``restore`` loads back.

        Raises
        ------
        NotFittedError
            If the scaler has not been fitted.
        """
        dimensions: CovariateDimensions = self.dimensions
        states: dict[str, ScalerState] = {}
        if dimensions.known_channels > 0:
            states[KNOWN_STATE] = self._known_scaler.to_state()
        if dimensions.static_channels > 0:
            states[STATIC_STATE] = self._static_scaler.to_state()
        return states

    def restore(
        self, dimensions: CovariateDimensions, cardinalities: tuple[int, ...], states: Mapping[str, ScalerState]
    ) -> None:
        """
        Load fitted statistics in place of fitting.

        Parameters
        ----------
        dimensions : CovariateDimensions
            The dimensions of the training covariates.
        cardinalities : tuple[int, ...]
            The number of categories of each categorical static feature.
        states : Mapping[str, ScalerState]
            The states exported by ``to_states``.

        Raises
        ------
        ValueError
            If the cardinalities, the states and the dimensions disagree.
        """
        if len(cardinalities) != dimensions.static_categories or any(count <= 0 for count in cardinalities):
            raise ValueError(
                f"{dimensions.static_categories} positive numbers of categories are required, got {cardinalities}"
            )
        expected: set[str] = {
            name
            for name, count in ((KNOWN_STATE, dimensions.known_channels), (STATIC_STATE, dimensions.static_channels))
            if count > 0
        }
        if set(states) != expected:
            raise ValueError(f"covariate scaler states {sorted(expected)} are required, got {sorted(states)}")
        if dimensions.known_channels > 0:
            self._known_scaler.restore_state(states[KNOWN_STATE], dimensions.known_channels)
        if dimensions.static_channels > 0:
            self._static_scaler.restore_state(states[STATIC_STATE], dimensions.static_channels)
        self._cardinalities = cardinalities
        self._dimensions = dimensions
