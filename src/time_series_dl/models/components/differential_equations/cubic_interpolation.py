from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .cubic_path import CubicPath
from .interval_differences import IntervalDifferences


class CubicInterpolation(ABC):
    """
    Piecewise cubic interpolation of knots, batched over samples of different lengths.

    Knots after a sample's length are padding and never change the path of its real knots.
    """

    def fit(self, times: Tensor, values: Tensor, lengths: Tensor) -> CubicPath:
        """
        Build the path through the knots.

        Parameters
        ----------
        times : Tensor
            Strictly increasing knot times of each sample with shape (batch, num_knots), padded after its length.
        values : Tensor
            Knot values with shape (batch, num_knots, channels).
        lengths : Tensor
            Number of knots of each sample with shape (batch,); at least 1.

        Returns
        -------
        CubicPath
            The path.

        Raises
        ------
        ValueError
            If the shapes disagree or a length is out of range.
        """
        if times.dim() != 2 or values.dim() != 3 or tuple(values.shape[:2]) != tuple(times.shape):
            raise ValueError(
                "times and values must have shapes (batch, num_knots) and (batch, num_knots, channels), got "
                + f"{tuple(times.shape)} and {tuple(values.shape)}"
            )
        if tuple(lengths.shape) != (times.shape[0],):
            raise ValueError(f"lengths must have shape ({times.shape[0]},), got {tuple(lengths.shape)}")
        if times.shape[1] == 0 or bool((lengths < 1).any()) or bool((lengths > times.shape[1]).any()):
            raise ValueError(f"lengths must be in [1, {times.shape[1]}]")
        if times.shape[1] == 1:
            # A single knot gets one padding interval, so every path has at least one interval.
            times = torch.cat([times, times], dim=1)
            values = torch.cat([values, values], dim=1)
        is_valid: Tensor = torch.arange(times.shape[1] - 1, device=times.device).unsqueeze(0) < (lengths - 1).unsqueeze(
            1
        )
        widths: Tensor = torch.where(is_valid, times[:, 1:] - times[:, :-1], torch.zeros_like(times[:, 1:]))
        differences: Tensor = torch.where(
            is_valid.unsqueeze(-1), values[:, 1:] - values[:, :-1], torch.zeros_like(values[:, 1:])
        )
        safe_widths: Tensor = torch.where(is_valid, widths, torch.ones_like(widths))
        intervals: IntervalDifferences = IntervalDifferences(
            widths=widths, differences=differences, slopes=differences / safe_widths.unsqueeze(-1)
        )
        return self._build_path(times, values, lengths, intervals)

    @abstractmethod
    def _build_path(self, times: Tensor, values: Tensor, lengths: Tensor, intervals: IntervalDifferences) -> CubicPath:
        """
        Compute the coefficients of every interval.

        Parameters
        ----------
        times : Tensor
            Knot times with shape (batch, num_knots), at least 2 knots.
        values : Tensor
            Knot values with shape (batch, num_knots, channels).
        lengths : Tensor
            Number of real knots of each sample with shape (batch,).
        intervals : IntervalDifferences
            Widths, differences and slopes of the intervals, zero on padding intervals.

        Returns
        -------
        CubicPath
            The path.
        """
