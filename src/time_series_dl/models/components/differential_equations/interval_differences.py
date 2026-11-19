from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class IntervalDifferences:
    """
    Widths and value differences of the intervals between consecutive knots, zero on padding intervals.

    Attributes
    ----------
    widths : Tensor
        Interval widths with shape (batch, num_intervals).
    differences : Tensor
        Value differences with shape (batch, num_intervals, channels).
    slopes : Tensor
        Differences divided by the widths, with the shape of ``differences``.
    """

    widths: Tensor
    differences: Tensor
    slopes: Tensor
