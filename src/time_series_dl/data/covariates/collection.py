from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray
from ..series_collection import SeriesCollection
from ..sliding_windows import SlidingWindows
from .dimensions import CovariateDimensions
from .mismatch_error import CovariateMismatchError
from .series_covariates import SeriesCovariates
from .window_covariates import WindowCovariates


@dataclass(frozen=True, eq=False)
class CovariateCollection:
    """
    Covariates of each series of a ``SeriesCollection``, with the same dimensions for every series.

    Attributes
    ----------
    covariates : tuple[SeriesCovariates, ...]
        The covariates of each series, in the order of the series.

    Raises
    ------
    ValueError
        If there are no covariates.
    CovariateMismatchError
        If the series have covariates of different dimensions.
    """

    covariates: tuple[SeriesCovariates, ...]

    def __post_init__(self) -> None:
        if not self.covariates:
            raise ValueError("at least one series' covariates are required.")
        dimensions: set[CovariateDimensions] = {covariates.dimensions for covariates in self.covariates}
        if len(dimensions) != 1:
            raise CovariateMismatchError(f"every series must have covariates of the same dimensions, got {dimensions}")

    @classmethod
    def from_input(cls, covariates: SeriesCovariates | Sequence[SeriesCovariates]) -> CovariateCollection:
        """
        Collect the covariates of one series or of several series.

        Parameters
        ----------
        covariates : SeriesCovariates | Sequence[SeriesCovariates]
            The covariates of one series, or of each series.

        Returns
        -------
        CovariateCollection
            The validated collection.
        """
        if isinstance(covariates, SeriesCovariates):
            return cls((covariates,))
        return cls(tuple(covariates))

    @property
    def dimensions(self) -> CovariateDimensions:
        """
        Get the number of covariates of each kind.

        Returns
        -------
        CovariateDimensions
            The dimensions shared by every series.
        """
        return self.covariates[0].dimensions

    def require_aligned(self, series: SeriesCollection) -> None:
        """
        Check that there are covariates for each series and that the known covariates are as long as their series.

        Parameters
        ----------
        series : SeriesCollection
            The series the covariates belong to.

        Raises
        ------
        CovariateMismatchError
            If the numbers of series differ or known covariates have another length than their series.
        """
        if len(self.covariates) != len(series.series):
            raise CovariateMismatchError(
                f"covariates of {len(series.series)} series are required, got {len(self.covariates)}"
            )
        for covariates, values in zip(self.covariates, series.series, strict=True):
            covariates.require_known_length(int(values.shape[0]))

    def extract_windows(self, windows: SlidingWindows, series: SeriesCollection) -> WindowCovariates:
        """
        Take the covariates of every window of every series, in the order of the windows of the series.

        Parameters
        ----------
        windows : SlidingWindows
            The windows taken from each series.
        series : SeriesCollection
            The series, each at least one window long.

        Returns
        -------
        WindowCovariates
            The covariates of every window.

        Raises
        ------
        CovariateMismatchError
            If the covariates are not aligned with the series.
        """
        self.require_aligned(series)
        return WindowCovariates.concatenate(
            [
                covariates.extract_windows(windows, int(values.shape[0]))
                for covariates, values in zip(self.covariates, series.series, strict=True)
            ]
        )

    def concatenate_known(self) -> FloatArray | None:
        """
        Stack every time step of the known covariates of every series.

        Returns
        -------
        FloatArray | None
            The known covariates with shape (total_length, known_channels); None when there are none.
        """
        known: list[FloatArray] = [covariates.known for covariates in self.covariates if covariates.known is not None]
        return np.concatenate(known, axis=0) if known else None

    def stack_static(self) -> FloatArray | None:
        """
        Stack the numeric static features of every series.

        Returns
        -------
        FloatArray | None
            The features with shape (num_series, static_channels); None when there are none.
        """
        static: list[FloatArray] = [
            covariates.static for covariates in self.covariates if covariates.static is not None
        ]
        return np.stack(static) if static else None

    def stack_static_categories(self) -> NDArray[np.integer] | None:
        """
        Stack the categorical static codes of every series.

        Returns
        -------
        NDArray[np.integer] | None
            The codes with shape (num_series, static_categories); None when there are none.
        """
        codes: list[NDArray[np.integer]] = [
            covariates.static_categories for covariates in self.covariates if covariates.static_categories is not None
        ]
        return np.stack(codes) if codes else None
