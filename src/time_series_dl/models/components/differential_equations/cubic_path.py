from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class CubicPath:
    """
    Piecewise cubic path through the knots of each sample, written per interval as a polynomial of the position
    ``s`` in [0, 1] within the interval: ``constant + linear * s + quadratic * s**2 + cubic * s**3``.

    Intervals at or after a sample's last knot are padding; their coefficients other than the constant are zero, so
    the path is flat there.

    Attributes
    ----------
    knot_times : Tensor
        Non-decreasing knot times with shape (batch, num_intervals + 1).
    lengths : Tensor
        Number of knots of each sample with shape (batch,); at least 1.
    constant : Tensor
        Constant coefficients with shape (batch, num_intervals, channels).
    linear : Tensor
        Linear coefficients with the same shape.
    quadratic : Tensor
        Quadratic coefficients with the same shape.
    cubic : Tensor
        Cubic coefficients with the same shape.

    Raises
    ------
    ValueError
        If the shapes disagree or there is no interval.
    """

    knot_times: Tensor
    lengths: Tensor
    constant: Tensor
    linear: Tensor
    quadratic: Tensor
    cubic: Tensor

    def __post_init__(self) -> None:
        if self.knot_times.dim() != 2 or self.knot_times.shape[1] < 2:
            raise ValueError(f"knot_times must have shape (batch, num_knots >= 2), got {tuple(self.knot_times.shape)}")
        expected: tuple[int, ...] = (self.knot_times.shape[0], self.knot_times.shape[1] - 1)
        for name, coefficients in (
            ("constant", self.constant),
            ("linear", self.linear),
            ("quadratic", self.quadratic),
            ("cubic", self.cubic),
        ):
            if coefficients.dim() != 3 or tuple(coefficients.shape[:2]) != expected:
                raise ValueError(f"{name} must have shape {(*expected, 'channels')}, got {tuple(coefficients.shape)}")
        if tuple(self.lengths.shape) != (self.knot_times.shape[0],):
            raise ValueError(f"lengths must have shape ({self.knot_times.shape[0]},), got {tuple(self.lengths.shape)}")

    @property
    def num_intervals(self) -> int:
        """
        Get the number of intervals, including padding.

        Returns
        -------
        int
            The number of intervals.
        """
        return int(self.constant.shape[1])

    def derivative(self, interval: int, position: Tensor) -> Tensor:
        """
        Get the derivative of the path with respect to the position within an interval.

        Parameters
        ----------
        interval : int
            Index of the interval.
        position : Tensor
            Position within the interval, broadcastable against shape (batch, channels), such as (batch, 1).

        Returns
        -------
        Tensor
            The derivative with shape (batch, channels).

        Raises
        ------
        IndexError
            If the interval does not exist.
        """
        if not 0 <= interval < self.num_intervals:
            raise IndexError(f"interval must be in [0, {self.num_intervals}), got {interval}")
        linear: Tensor = self.linear[:, interval]
        quadratic: Tensor = self.quadratic[:, interval]
        cubic: Tensor = self.cubic[:, interval]
        return linear + position * (2.0 * quadratic + 3.0 * position * cubic)

    def evaluate(self, times: Tensor) -> Tensor:
        """
        Evaluate the path at given times; beyond the first and last knots it extends the outer cubic pieces.

        Parameters
        ----------
        times : Tensor
            Times of each sample with shape (batch, num_times).

        Returns
        -------
        Tensor
            The values with shape (batch, num_times, channels).
        """
        num_intervals: int = self.num_intervals
        last_interval: Tensor = (self.lengths - 2).clamp(min=0).unsqueeze(1)
        interval: Tensor = torch.searchsorted(self.knot_times.contiguous(), times.contiguous(), right=True) - 1
        interval = torch.minimum(interval.clamp(min=0, max=num_intervals - 1), last_interval)
        widths: Tensor = self.knot_times[:, 1:] - self.knot_times[:, :-1]
        start: Tensor = self.knot_times.gather(1, interval)
        width: Tensor = widths.gather(1, interval)
        position: Tensor = ((times - start) / torch.where(width > 0, width, torch.ones_like(width))).unsqueeze(-1)
        index: Tensor = interval.unsqueeze(-1).expand(-1, -1, self.constant.shape[2])
        constant: Tensor = self.constant.gather(1, index)
        linear: Tensor = self.linear.gather(1, index)
        quadratic: Tensor = self.quadratic.gather(1, index)
        cubic: Tensor = self.cubic.gather(1, index)
        return constant + position * (linear + position * (quadratic + position * cubic))
