from enum import Enum
from typing import cast

import torch
from torch import Tensor


class Discretization(Enum):
    """
    Conversion of a continuous-time state space model to a recurrence over time steps.

    Attributes
    ----------
    ZERO_ORDER_HOLD
        Exact for inputs held constant over each step.
    BILINEAR
        Trapezoidal rule (Tustin), which maps the stable half-plane onto the unit disk.
    """

    ZERO_ORDER_HOLD = "zero_order_hold"
    BILINEAR = "bilinear"

    def discretize_dense(self, transition: Tensor, input_matrix: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
        """
        Discretize a state space model with dense matrices.

        Parameters
        ----------
        transition : Tensor
            Continuous state matrix with shape (state_size, state_size).
        input_matrix : Tensor
            Continuous input matrix with shape (state_size, input_size).
        step : Tensor
            Step sizes with any shape (...).

        Returns
        -------
        tuple[Tensor, Tensor]
            The discrete state matrices with shape (..., state_size, state_size) and the discrete input matrices with
            shape (..., state_size, input_size).
        """
        state_size: int = transition.shape[0]
        steps: Tensor = step[..., None, None]
        match self:
            case Discretization.ZERO_ORDER_HOLD:
                # exp([[A, B], [0, 0]] dt) holds exp(A dt) and A^-1 (exp(A dt) - I) B without inverting A.
                augmented: Tensor = torch.zeros(
                    (*step.shape, state_size + input_matrix.shape[1], state_size + input_matrix.shape[1]),
                    dtype=transition.dtype,
                )
                augmented[..., :state_size, :state_size] = transition * steps
                augmented[..., :state_size, state_size:] = input_matrix * steps
                exponential: Tensor = cast(Tensor, torch.linalg.matrix_exp(augmented))
                return exponential[..., :state_size, :state_size], exponential[..., :state_size, state_size:]
            case Discretization.BILINEAR:
                identity: Tensor = torch.eye(state_size, dtype=transition.dtype)
                backward: Tensor = identity - transition * steps / 2.0
                forward: Tensor = identity + transition * steps / 2.0
                return (
                    cast(Tensor, torch.linalg.solve(backward, forward)),
                    cast(Tensor, torch.linalg.solve(backward, input_matrix * steps)),
                )

    def discretize_diagonal(self, eigenvalues: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
        """
        Discretize a state space model with a diagonal state matrix.

        Parameters
        ----------
        eigenvalues : Tensor
            Complex diagonal of the continuous state matrix.
        step : Tensor
            Step sizes, broadcastable with ``eigenvalues``.

        Returns
        -------
        tuple[Tensor, Tensor]
            The discrete diagonal state matrix, and the factor by which the continuous input matrix is multiplied.
        """
        scaled: Tensor = eigenvalues * step
        match self:
            case Discretization.ZERO_ORDER_HOLD:
                decay: Tensor = torch.exp(scaled)
                return decay, (decay - 1.0) / eigenvalues
            case Discretization.BILINEAR:
                backward: Tensor = 1.0 - scaled / 2.0
                return (1.0 + scaled / 2.0) / backward, step / backward
