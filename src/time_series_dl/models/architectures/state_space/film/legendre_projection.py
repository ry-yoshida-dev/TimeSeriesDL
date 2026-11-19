import numpy as np
import torch
from scipy.special import eval_legendre
from torch import Tensor, nn

from ....components.state_space import Discretization, FFTConvolution, HiPPOMeasure, HiPPOMemory


class FiLMLegendreProjection(nn.Module):
    """
    Fixed Legendre projection of FiLM: a HiPPO-LegT memory of a window run over the input, and the Legendre
    evaluation that reconstructs the newest steps of the window from a memory state.

    Parameters
    ----------
    state_size : int
        Number of Legendre polynomials.
    window_length : int
        Number of steps of the window the memory covers.
    input_length : int
        Number of time steps of the input; the memory reads its last min(window_length, input_length) steps.
    horizon : int
        Number of newest steps of the window that are reconstructed; at most window_length.

    Raises
    ------
    ValueError
        If a size is not positive or horizon exceeds window_length.
    """

    def __init__(self, state_size: int, window_length: int, input_length: int, horizon: int) -> None:
        for name, value in (
            ("state_size", state_size),
            ("window_length", window_length),
            ("input_length", input_length),
            ("horizon", horizon),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if horizon > window_length:
            raise ValueError(f"horizon {horizon} must not exceed window_length {window_length}")
        super().__init__()
        self.read_length: int = min(window_length, input_length)
        memory: HiPPOMemory = HiPPOMemory(
            HiPPOMeasure.LEGT, state_size, 1, float(window_length), Discretization.BILINEAR
        )
        self.kernel: Tensor
        self.register_buffer("kernel", memory.impulse_response(self.read_length).T.contiguous())
        # The memory reconstructs the window as sum_n c_n P_n(1 - 2 r), with r from the oldest step (0) to the newest
        # (1); each step is evaluated at its centre.
        positions: np.ndarray = (np.arange(window_length - horizon, window_length) + 0.5) / window_length
        evaluation: np.ndarray = eval_legendre(np.arange(state_size)[:, None], 1.0 - 2.0 * positions[None, :])
        self.evaluation: Tensor
        self.register_buffer("evaluation", torch.tensor(evaluation, dtype=torch.float32))

    @property
    def state_size(self) -> int:
        """
        Get the number of Legendre polynomials.

        Returns
        -------
        int
            The size of each memory state.
        """
        return self.kernel.shape[0]

    def encode(self, values: Tensor) -> Tensor:
        """
        Run the memory over the last steps of each series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, channels, input_length).

        Returns
        -------
        Tensor
            Memory states with shape (batch, channels, state_size, read_length).
        """
        window: Tensor = values[..., -self.read_length :]
        return FFTConvolution.apply(window[:, :, None, :], self.kernel)

    def reconstruct(self, states: Tensor) -> Tensor:
        """
        Evaluate the newest steps of the window that a memory state stands for.

        Parameters
        ----------
        states : Tensor
            Memory states with shape (..., state_size).

        Returns
        -------
        Tensor
            Values with shape (..., horizon).
        """
        reconstructed: Tensor = states @ self.evaluation
        return reconstructed
