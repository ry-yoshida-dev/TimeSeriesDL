from enum import Enum

from torch import Tensor


class DiffusionTSLossKind(Enum):
    """
    Error of each value in the Diffusion-TS objective, in the time and in the frequency domain.

    Attributes
    ----------
    ABSOLUTE
        Absolute error (L1), the authors' setting.
    SQUARED
        Squared error (L2).
    """

    ABSOLUTE = "absolute"
    SQUARED = "squared"

    def compute(self, difference: Tensor) -> Tensor:
        """
        Compute the error of each value from its difference to the target.

        Parameters
        ----------
        difference : Tensor
            Prediction minus target, of any shape.

        Returns
        -------
        Tensor
            The errors, with the shape of the difference.
        """
        match self:
            case DiffusionTSLossKind.ABSOLUTE:
                return difference.abs()
            case DiffusionTSLossKind.SQUARED:
                return difference.square()
