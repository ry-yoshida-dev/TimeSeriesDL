from enum import Enum

from .cubic_interpolation import CubicInterpolation
from .cubic_interpolations import HermiteCubicInterpolation, NaturalCubicInterpolation


class CubicInterpolationKind(Enum):
    """
    Piecewise cubic interpolation of the knots of a control path.

    Attributes
    ----------
    NATURAL
        Natural cubic spline, twice continuously differentiable but not causal.
    HERMITE
        Hermite cubic with backward differences, continuously differentiable and causal.
    """

    NATURAL = "natural"
    HERMITE = "hermite"

    def build(self) -> CubicInterpolation:
        """
        Build the interpolation.

        Returns
        -------
        CubicInterpolation
            The interpolation.
        """
        match self:
            case CubicInterpolationKind.NATURAL:
                return NaturalCubicInterpolation()
            case CubicInterpolationKind.HERMITE:
                return HermiteCubicInterpolation()
