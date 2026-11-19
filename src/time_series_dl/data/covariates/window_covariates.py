from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray
from .array_validation import CovariateArrayValidation
from .dimensions import CovariateDimensions
from .mismatch_error import CovariateMismatchError


@dataclass(frozen=True, kw_only=True, eq=False)
class WindowCovariates:
    """
    Covariates of a batch of forecast windows, in the order of the windows.

    Attributes
    ----------
    known : FloatArray | None
        Known covariates with shape (num_windows, input_length + horizon, known_channels): the past window of each
        forecast followed by its horizon. None when there are no known covariates.
    static : FloatArray | None
        Numeric static features of each window's series with shape (num_windows, static_channels).
    static_categories : NDArray[np.integer] | None
        Integer codes of the categorical static features of each window's series with shape
        (num_windows, static_categories).

    Raises
    ------
    ValueError
        If no covariate is given, an array is invalid, or the numbers of windows differ.
    TypeError
        If an array has the wrong dtype.
    """

    known: FloatArray | None = None
    static: FloatArray | None = None
    static_categories: NDArray[np.integer] | None = None

    def __post_init__(self) -> None:
        if self.known is None and self.static is None and self.static_categories is None:
            raise ValueError("at least one of known, static and static_categories is required.")
        counts: set[int] = set()
        if self.known is not None:
            known: FloatArray = CovariateArrayValidation.require_real(self.known, "known", 3)
            object.__setattr__(self, "known", known)
            counts.add(known.shape[0])
        if self.static is not None:
            static: FloatArray = CovariateArrayValidation.require_real(self.static, "static", 2)
            object.__setattr__(self, "static", static)
            counts.add(static.shape[0])
        if self.static_categories is not None:
            codes: NDArray[np.int64] = CovariateArrayValidation.require_codes(
                self.static_categories, "static_categories", 2
            )
            object.__setattr__(self, "static_categories", codes)
            counts.add(codes.shape[0])
        if len(counts) != 1:
            raise ValueError(f"every covariate array must have the same number of windows, got {sorted(counts)}")

    @staticmethod
    def concatenate(parts: Sequence[WindowCovariates]) -> WindowCovariates:
        """
        Join the windows of several batches.

        Parameters
        ----------
        parts : Sequence[WindowCovariates]
            Batches with the same dimensions and known length.

        Returns
        -------
        WindowCovariates
            The windows of every batch, in order.

        Raises
        ------
        ValueError
            If there is no batch.
        CovariateMismatchError
            If the batches have different dimensions or known lengths.
        """
        if not parts:
            raise ValueError("at least one batch of window covariates is required.")
        first: WindowCovariates = parts[0]
        if any(part.dimensions != first.dimensions or part.known_length != first.known_length for part in parts[1:]):
            raise CovariateMismatchError("window covariates to join must have the same dimensions and length.")
        return WindowCovariates(
            known=None if first.known is None else np.concatenate([part.require_known() for part in parts]),
            static=None if first.static is None else np.concatenate([part.require_static() for part in parts]),
            static_categories=(
                None
                if first.static_categories is None
                else np.concatenate([part.require_static_categories() for part in parts])
            ),
        )

    @property
    def num_windows(self) -> int:
        """
        Get the number of windows.

        Returns
        -------
        int
            The number of windows.
        """
        for array in (self.known, self.static, self.static_categories):
            if array is not None:
                return int(array.shape[0])
        raise AssertionError("validated covariates hold at least one array.")

    @property
    def known_length(self) -> int | None:
        """
        Get the number of time steps of the known covariates of each window.

        Returns
        -------
        int | None
            The length; None when there are no known covariates.
        """
        return None if self.known is None else int(self.known.shape[1])

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
            known_channels=0 if self.known is None else int(self.known.shape[2]),
            static_channels=0 if self.static is None else int(self.static.shape[1]),
            static_categories=0 if self.static_categories is None else int(self.static_categories.shape[1]),
        )

    def require_known(self) -> FloatArray:
        """
        Get the known covariates, which must be present.

        Returns
        -------
        FloatArray
            The known covariates.

        Raises
        ------
        CovariateMismatchError
            If there are no known covariates.
        """
        if self.known is None:
            raise CovariateMismatchError("known covariates are required.")
        return self.known

    def require_static(self) -> FloatArray:
        """
        Get the numeric static features, which must be present.

        Returns
        -------
        FloatArray
            The features.

        Raises
        ------
        CovariateMismatchError
            If there are no numeric static features.
        """
        if self.static is None:
            raise CovariateMismatchError("numeric static covariates are required.")
        return self.static

    def require_static_categories(self) -> NDArray[np.integer]:
        """
        Get the categorical static codes, which must be present.

        Returns
        -------
        NDArray[np.integer]
            The codes.

        Raises
        ------
        CovariateMismatchError
            If there are no categorical static features.
        """
        if self.static_categories is None:
            raise CovariateMismatchError("categorical static covariates are required.")
        return self.static_categories
