from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray
from ..sliding_windows import SlidingWindows
from .array_validation import CovariateArrayValidation
from .dimensions import CovariateDimensions
from .mismatch_error import CovariateMismatchError
from .window_covariates import WindowCovariates


@dataclass(frozen=True, kw_only=True, eq=False)
class SeriesCovariates:
    """
    Covariates of one series, given next to the series to ``Forecaster``.

    Attributes
    ----------
    known : FloatArray | None
        Covariates known in advance, such as calendar features or planned promotions, with shape
        (length, known_channels). Step ``t`` is aligned with step ``t`` of the series; for ``Forecaster.forecast`` the
        array extends ``horizon`` steps past the end of the series. None when there are none.
    static : FloatArray | None
        Numeric features of the series that do not change over time, such as a store's size, with shape
        (static_channels,).
    static_categories : NDArray[np.integer] | None
        Integer codes of categorical features of the series, such as a store id, with shape (static_categories,);
        each code is an index in ``[0, number of categories)``.

    Raises
    ------
    ValueError
        If no covariate is given or an array is invalid.
    TypeError
        If an array has the wrong dtype.
    """

    known: FloatArray | None = None
    static: FloatArray | None = None
    static_categories: NDArray[np.integer] | None = None

    def __post_init__(self) -> None:
        if self.known is None and self.static is None and self.static_categories is None:
            raise ValueError("at least one of known, static and static_categories is required.")
        if self.known is not None:
            object.__setattr__(self, "known", CovariateArrayValidation.require_real(self.known, "known", 2))
        if self.static is not None:
            object.__setattr__(self, "static", CovariateArrayValidation.require_real(self.static, "static", 1))
        if self.static_categories is not None:
            object.__setattr__(
                self,
                "static_categories",
                CovariateArrayValidation.require_codes(self.static_categories, "static_categories", 1),
            )

    @property
    def dimensions(self) -> CovariateDimensions:
        """
        Get the number of covariates of each kind.

        Returns
        -------
        CovariateDimensions
            The dimensions.
        """
        return CovariateDimensions(
            known_channels=0 if self.known is None else int(self.known.shape[1]),
            static_channels=0 if self.static is None else int(self.static.shape[0]),
            static_categories=0 if self.static_categories is None else int(self.static_categories.shape[0]),
        )

    def require_known_length(self, length: int) -> None:
        """
        Check that the known covariates, if any, have the given number of steps.

        Parameters
        ----------
        length : int
            The required number of steps.

        Raises
        ------
        CovariateMismatchError
            If the known covariates have another length.
        """
        if self.known is not None and self.known.shape[0] != length:
            raise CovariateMismatchError(f"known covariates must have {length} steps, got {self.known.shape[0]}")

    def extract_windows(self, windows: SlidingWindows, series_length: int) -> WindowCovariates:
        """
        Take the covariates of every window of the series, aligned with ``windows.extract`` of the series.

        Parameters
        ----------
        windows : SlidingWindows
            The windows taken from the series.
        series_length : int
            Number of steps of the series, at least one window long.

        Returns
        -------
        WindowCovariates
            The known covariates of each window and the static features repeated for each window.

        Raises
        ------
        CovariateMismatchError
            If the known covariates are not as long as the series.
        """
        self.require_known_length(series_length)
        count: int = int(windows.starts(series_length).size)
        return WindowCovariates(
            known=None if self.known is None else windows.extract(self.known),
            static=None if self.static is None else np.repeat(self.static[np.newaxis], count, axis=0),
            static_categories=(
                None if self.static_categories is None else np.repeat(self.static_categories[np.newaxis], count, axis=0)
            ),
        )

    def slice_window(self, start: int, length: int) -> WindowCovariates:
        """
        Take the covariates of one window.

        Parameters
        ----------
        start : int
            First step of the window.
        length : int
            Number of steps of the window.

        Returns
        -------
        WindowCovariates
            The covariates as a batch of one window.

        Raises
        ------
        CovariateMismatchError
            If the known covariates do not cover the window.
        """
        if self.known is not None and (start < 0 or start + length > self.known.shape[0]):
            raise CovariateMismatchError(
                f"known covariates of {self.known.shape[0]} steps do not cover steps [{start}, {start + length})"
            )
        return WindowCovariates(
            known=None if self.known is None else self.known[np.newaxis, start : start + length],
            static=None if self.static is None else self.static[np.newaxis],
            static_categories=None if self.static_categories is None else self.static_categories[np.newaxis],
        )
