from enum import Enum

import torch
from torch import Tensor


class HiPPOMeasure(Enum):
    """
    Measure over the history that a HiPPO memory (Gu et al., 2020) projects the input onto, which fixes its continuous
    state matrix A and input matrix B.

    Attributes
    ----------
    LEGS
        Scaled Legendre: uniform weight over the whole history, so the dynamics slow down as time passes.
    LEGT
        Translated Legendre: uniform weight over a sliding window, as in the Legendre Memory Unit (Voelker et al.,
        2019); the state at time t reconstructs the window through Legendre polynomials.
    LAGT
        Translated Laguerre: exponentially decaying weight over the past.
    FOUT
        Translated Fourier: Fourier basis over a sliding window.
    """

    LEGS = "legs"
    LEGT = "legt"
    LAGT = "lagt"
    FOUT = "fout"

    @property
    def is_time_invariant(self) -> bool:
        """
        Check whether the dynamics are the same at every step.

        Returns
        -------
        bool
            False for the scaled measure, whose dynamics are divided by the elapsed time.
        """
        return self is not HiPPOMeasure.LEGS

    def transition(self, state_size: int) -> tuple[Tensor, Tensor]:
        """
        Build the continuous state and input matrices, in float64.

        Translated measures describe a window of length 1, and the scaled measure the history up to time 1.

        Parameters
        ----------
        state_size : int
            Number of basis functions.

        Returns
        -------
        tuple[Tensor, Tensor]
            A with shape (state_size, state_size) and B with shape (state_size, 1).

        Raises
        ------
        ValueError
            If state_size is not positive, or is odd for the Fourier measure.
        """
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        order: Tensor = torch.arange(state_size, dtype=torch.float64)
        row: Tensor = order[:, None]
        column: Tensor = order[None, :]
        match self:
            case HiPPOMeasure.LEGS:
                scale: Tensor = torch.sqrt(2.0 * order + 1.0)
                lower: Tensor = torch.where(row > column, scale[:, None] * scale[None, :], 0.0)
                return -(lower + torch.diag(order + 1.0)), scale[:, None]
            case HiPPOMeasure.LEGT:
                # The unnormalized form of the Legendre Memory Unit, whose state reconstructs the window as
                # sum_n c_n P_n(1 - 2 r) at the relative position r in [0, 1] from oldest to newest.
                weight: Tensor = 2.0 * order + 1.0
                sign: Tensor = torch.where(row < column, -1.0, (-1.0) ** (row - column + 1.0))
                return weight[:, None] * sign, (weight * (-1.0) ** order)[:, None]
            case HiPPOMeasure.LAGT:
                transition: Tensor = torch.eye(state_size, dtype=torch.float64) / 2.0 - torch.tril(
                    torch.ones(state_size, state_size, dtype=torch.float64)
                )
                return transition, torch.ones(state_size, 1, dtype=torch.float64)
            case HiPPOMeasure.FOUT:
                if state_size % 2 != 0:
                    raise ValueError(f"the Fourier measure needs an even state_size, got {state_size}")
                frequencies: Tensor = torch.arange(state_size // 2, dtype=torch.float64)
                rates: Tensor = torch.stack([torch.zeros_like(frequencies), frequencies], dim=-1).reshape(-1)[1:]
                rotation: Tensor = torch.pi * (torch.diag(rates, -1) - torch.diag(rates, 1))
                input_vector: Tensor = torch.zeros(state_size, dtype=torch.float64)
                input_vector[0::2] = 2.0**0.5
                input_vector[0] = 1.0
                return rotation - input_vector[:, None] * input_vector[None, :], input_vector[:, None]
