from typing import override

import torch
from torch import Tensor

from ..cubic_interpolation import CubicInterpolation
from ..cubic_path import CubicPath
from ..interval_differences import IntervalDifferences


class HermiteCubicInterpolation(CubicInterpolation):
    """
    Hermite cubic interpolation with backward differences (Morrill et al., 2021, as in ``torchcde``): each interval
    starts with the slope of the previous interval and ends with its own slope, so the path is continuously
    differentiable and causal, since the path up to a knot depends only on the knots up to it.
    """

    @override
    def _build_path(self, times: Tensor, values: Tensor, lengths: Tensor, intervals: IntervalDifferences) -> CubicPath:
        previous_slopes: Tensor = torch.cat([intervals.slopes[:, :1], intervals.slopes[:, :-1]], dim=1)
        start_derivative: Tensor = intervals.widths.unsqueeze(-1) * previous_slopes
        end_derivative: Tensor = intervals.differences
        return CubicPath(
            knot_times=times,
            lengths=lengths,
            constant=values[:, :-1],
            linear=start_derivative,
            quadratic=3.0 * intervals.differences - 2.0 * start_derivative - end_derivative,
            cubic=start_derivative + end_derivative - 2.0 * intervals.differences,
        )
