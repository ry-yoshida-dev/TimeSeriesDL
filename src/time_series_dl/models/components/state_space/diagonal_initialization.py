from enum import Enum

import torch
from torch import Tensor

from .normal_plus_low_rank import NormalPlusLowRank


class DiagonalInitialization(Enum):
    """
    Initial eigenvalues of a diagonal state space model (Gu et al., 2022, "On the Parameterization and
    Initialization of Diagonal State Space Models").

    Attributes
    ----------
    LEGS
        S4D-LegS: the eigenvalues of the normal part of HiPPO-LegS, which approximate S4.
    INVERSE
        S4D-Inv: real part -1/2 and imaginary parts that fall off as the inverse of the index.
    LINEAR
        S4D-Lin: real part -1/2 and imaginary parts pi n, the frequencies of a Fourier basis.
    """

    LEGS = "legs"
    INVERSE = "inverse"
    LINEAR = "linear"

    def eigenvalues(self, num_modes: int) -> Tensor:
        """
        Build the eigenvalues with positive imaginary parts, each standing for itself and its conjugate.

        Parameters
        ----------
        num_modes : int
            Number of conjugate pairs, half the state size.

        Returns
        -------
        Tensor
            The complex64 eigenvalues with shape (num_modes,).
        """
        index: Tensor = torch.arange(num_modes, dtype=torch.float64)
        state_size: int = 2 * num_modes
        match self:
            case DiagonalInitialization.LEGS:
                eigenvalues: Tensor = NormalPlusLowRank.from_legs(state_size).eigenvalues[num_modes:]
                return eigenvalues.to(torch.complex64)
            case DiagonalInitialization.INVERSE:
                imaginary: Tensor = state_size / torch.pi * (state_size / (2.0 * index + 1.0) - 1.0)
            case DiagonalInitialization.LINEAR:
                imaginary = torch.pi * index
        return torch.complex(torch.full_like(imaginary, -0.5), imaginary).to(torch.complex64)
