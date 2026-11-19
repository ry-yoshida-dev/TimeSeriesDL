from enum import Enum

import torch
from torch import Tensor


class NoiseScheduleKind(Enum):
    """
    Shape of the noise variances of a diffusion process over its steps.

    Attributes
    ----------
    LINEAR
        Variances evenly spaced from the first to the last (Ho et al., 2020; TimeGrad).
    QUADRATIC
        Square roots of the variances evenly spaced, so the variances grow quadratically (CSDI).
    """

    LINEAR = "linear"
    QUADRATIC = "quadratic"

    def compute_variances(self, num_steps: int, first: float, last: float) -> Tensor:
        """
        Compute the noise variance of every diffusion step.

        Parameters
        ----------
        num_steps : int
            Number of diffusion steps.
        first : float
            Variance of the first step.
        last : float
            Variance of the last step.

        Returns
        -------
        Tensor
            The variances in double precision with shape (num_steps,).
        """
        match self:
            case NoiseScheduleKind.LINEAR:
                return torch.linspace(first, last, num_steps, dtype=torch.float64)
            case NoiseScheduleKind.QUADRATIC:
                return torch.linspace(first**0.5, last**0.5, num_steps, dtype=torch.float64).square()
