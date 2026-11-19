from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..cubic_interpolation import CubicInterpolation
from ..cubic_path import CubicPath
from ..interval_differences import IntervalDifferences


class NaturalCubicInterpolation(CubicInterpolation):
    """
    Natural cubic spline: twice continuously differentiable, with zero second derivative at each sample's first and
    last knots. Every knot influences the whole path, so the path is not causal.

    The tridiagonal system of the second derivatives is solved by the Thomas algorithm, with identity rows at the
    end knots and the padding knots, so padding never changes the path of the real knots.
    """

    @override
    def _build_path(self, times: Tensor, values: Tensor, lengths: Tensor, intervals: IntervalDifferences) -> CubicPath:
        num_knots: int = times.shape[1]
        knots: Tensor = torch.arange(num_knots, device=times.device).unsqueeze(0)
        is_interior: Tensor = (knots >= 1) & (knots <= (lengths - 2).unsqueeze(1))
        previous_widths: Tensor = functional.pad(intervals.widths, (1, 0))
        next_widths: Tensor = functional.pad(intervals.widths, (0, 1))
        previous_slopes: Tensor = functional.pad(intervals.slopes, (0, 0, 1, 0))
        next_slopes: Tensor = functional.pad(intervals.slopes, (0, 0, 0, 1))
        zeros: Tensor = torch.zeros_like(previous_widths)
        lower: Tensor = torch.where(is_interior, previous_widths, zeros)
        upper: Tensor = torch.where(is_interior, next_widths, zeros)
        diagonal: Tensor = torch.where(is_interior, 2.0 * (previous_widths + next_widths), torch.ones_like(zeros))
        right_side: Tensor = torch.where(
            is_interior.unsqueeze(-1), 6.0 * (next_slopes - previous_slopes), torch.zeros_like(next_slopes)
        )
        second_derivatives: Tensor = self._solve_tridiagonal(lower, diagonal, upper, right_side)
        squared_widths: Tensor = intervals.widths.square().unsqueeze(-1)
        start_curvature: Tensor = second_derivatives[:, :-1]
        end_curvature: Tensor = second_derivatives[:, 1:]
        return CubicPath(
            knot_times=times,
            lengths=lengths,
            constant=values[:, :-1],
            linear=intervals.differences - squared_widths * (2.0 * start_curvature + end_curvature) / 6.0,
            quadratic=squared_widths * start_curvature / 2.0,
            cubic=squared_widths * (end_curvature - start_curvature) / 6.0,
        )

    @staticmethod
    def _solve_tridiagonal(lower: Tensor, diagonal: Tensor, upper: Tensor, right_side: Tensor) -> Tensor:
        # The sweeps are sequential by nature; each step is vectorized over the batch and the channels.
        num_knots: int = diagonal.shape[1]
        upper_factors: list[Tensor] = []
        solutions: list[Tensor] = []
        for knot in range(num_knots):
            pivot: Tensor = diagonal[:, knot]
            right: Tensor = right_side[:, knot]
            if knot > 0:
                pivot = pivot - lower[:, knot] * upper_factors[-1]
                right = right - lower[:, knot].unsqueeze(-1) * solutions[-1]
            upper_factors.append(upper[:, knot] / pivot)
            solutions.append(right / pivot.unsqueeze(-1))
        result: list[Tensor] = [solutions[-1]]
        for knot in range(num_knots - 2, -1, -1):
            result.append(solutions[knot] - upper_factors[knot].unsqueeze(-1) * result[-1])
        return torch.stack(result[::-1], dim=1)
