from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .conformity_score import ConformityScore

if TYPE_CHECKING:
    from ..forecaster import Forecaster
    from .forecaster import ConformalForecaster


@dataclass(frozen=True, kw_only=True)
class ConformalSettings:
    """
    How conformal prediction calibrates forecast intervals.

    Attributes
    ----------
    coverage_levels : tuple[float, ...]
        Nominal coverages of the central intervals, strictly increasing in (0, 1).
    score : ConformityScore
        The conformity score the intervals are calibrated on.
    adaptation_rate : float
        Step size ``gamma`` of adaptive conformal inference (Gibbs and Candès, 2021), by which each observed miss or
        hit moves the miscoverage rate; the paper uses 0.005. It only matters when intervals are updated online.

    Raises
    ------
    ValueError
        If coverage_levels is empty, out of (0, 1) or not strictly increasing, or adaptation_rate is negative.
    """

    coverage_levels: tuple[float, ...] = (0.9,)
    score: ConformityScore = ConformityScore.ABSOLUTE_RESIDUAL
    adaptation_rate: float = 0.005

    def __post_init__(self) -> None:
        if not self.coverage_levels:
            raise ValueError("coverage_levels must not be empty.")
        if any(not 0.0 < level < 1.0 for level in self.coverage_levels):
            raise ValueError(f"coverage_levels must be in (0, 1), got {self.coverage_levels}")
        if any(lower >= upper for lower, upper in zip(self.coverage_levels, self.coverage_levels[1:], strict=False)):
            raise ValueError(f"coverage_levels must be strictly increasing, got {self.coverage_levels}")
        if self.adaptation_rate < 0.0:
            raise ValueError(f"adaptation_rate must not be negative, got {self.adaptation_rate}")

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """
        Get the levels of the interval bounds, as quantile levels of a forecast.

        Returns
        -------
        tuple[float, ...]
            The lower bound levels ``(1 - coverage) / 2`` and the upper ones ``(1 + coverage) / 2``, increasing.
        """
        lower: list[float] = [(1.0 - coverage) / 2.0 for coverage in reversed(self.coverage_levels)]
        upper: list[float] = [(1.0 + coverage) / 2.0 for coverage in self.coverage_levels]
        return (*lower, *upper)

    def build(self, forecaster: Forecaster) -> ConformalForecaster:
        """
        Build an uncalibrated conformal forecaster around a forecaster.

        Parameters
        ----------
        forecaster : Forecaster
            A forecaster, fitted before calibration.

        Returns
        -------
        ConformalForecaster
            The conformal forecaster.
        """
        from .forecaster import ConformalForecaster

        return ConformalForecaster(forecaster, self)
