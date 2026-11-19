import torch
from torch import Tensor, nn

from .discretization import Discretization
from .hippo_measure import HiPPOMeasure


class HiPPOMemory(nn.Module):
    """
    Fixed discrete HiPPO recurrence c_t = A_t c_{t-1} + B_t f_t, which keeps the projection of a scalar input
    history onto the basis of a measure.

    Parameters
    ----------
    measure : HiPPOMeasure
        The measure, which fixes the continuous dynamics.
    state_size : int
        Number of basis functions.
    length : int
        Number of steps the memory is run for; the scaled measure has different dynamics at each step.
    window_length : float
        Number of steps a translated measure covers; ignored by the scaled measure, whose step at time t is 1 / t.
    discretization : Discretization
        Conversion of the continuous dynamics to steps.

    Raises
    ------
    ValueError
        If length or window_length is not positive.
    """

    def __init__(
        self,
        measure: HiPPOMeasure,
        state_size: int,
        length: int,
        window_length: float,
        discretization: Discretization,
    ) -> None:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if window_length <= 0.0:
            raise ValueError(f"window_length must be positive, got {window_length}")
        super().__init__()
        self.is_time_invariant: bool = measure.is_time_invariant
        transition, input_matrix = measure.transition(state_size)
        steps: Tensor = (
            torch.tensor([1.0 / window_length], dtype=torch.float64)
            if self.is_time_invariant
            else 1.0 / torch.arange(1, length + 1, dtype=torch.float64)
        )
        transitions, input_vectors = discretization.discretize_dense(transition, input_matrix, steps)
        self.transitions: Tensor
        self.input_vectors: Tensor
        self.register_buffer("transitions", transitions.float())
        self.register_buffer("input_vectors", input_vectors.squeeze(-1).float())

    @property
    def state_size(self) -> int:
        """
        Get the number of basis functions.

        Returns
        -------
        int
            The size of the memory state.
        """
        return self.input_vectors.shape[-1]

    def step(self, memory: Tensor, inputs: Tensor, index: int) -> Tensor:
        """
        Advance the memory by one step.

        Parameters
        ----------
        memory : Tensor
            Memory with shape (..., state_size) before the step.
        inputs : Tensor
            Scalar inputs of the step with shape (...).
        index : int
            Zero-based index of the step, which selects the dynamics of the scaled measure.

        Returns
        -------
        Tensor
            The memory after the step, with the shape of ``memory``.
        """
        position: int = 0 if self.is_time_invariant else min(index, self.transitions.shape[0] - 1)
        advanced: Tensor = memory @ self.transitions[position].T + inputs[..., None] * self.input_vectors[position]
        return advanced

    def impulse_response(self, length: int) -> Tensor:
        """
        Compute the response A^k B of a time-invariant memory to a unit input k steps ago.

        The state after any input sequence is the causal convolution of the input with this response.

        Parameters
        ----------
        length : int
            Number of lags.

        Returns
        -------
        Tensor
            The response with shape (length, state_size).

        Raises
        ------
        ValueError
            If length is not positive or the measure is not time-invariant.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if not self.is_time_invariant:
            raise ValueError("only a time-invariant memory has an impulse response independent of time.")
        transition: Tensor = self.transitions[0].double()
        response: Tensor = self.input_vectors[0].double()
        responses: list[Tensor] = [response]
        for _ in range(length - 1):
            response = transition @ response
            responses.append(response)
        return torch.stack(responses).float()
