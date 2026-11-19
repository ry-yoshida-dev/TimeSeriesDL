import torch
from torch import Tensor


class LinearRecurrence:
    """
    Parallel evaluation of the diagonal linear recurrence h_t = a_t * h_{t-1} + b_t from h_0 = 0.

    The recurrence is associative, so it is evaluated in log2(length) vectorized steps (Hillis and Steele) instead of
    one step per time step; real and complex values are supported.
    """

    @staticmethod
    def scan(decays: Tensor, inputs: Tensor) -> Tensor:
        """
        Evaluate the recurrence along the second dimension.

        Parameters
        ----------
        decays : Tensor
            Factors a_t, broadcastable to the shape of ``inputs``.
        inputs : Tensor
            Inputs b_t with shape (batch, length, ...).

        Returns
        -------
        Tensor
            States h_t with the shape of ``inputs``.

        Raises
        ------
        ValueError
            If inputs has fewer than two dimensions.
        """
        if inputs.dim() < 2:
            raise ValueError(f"inputs must have a batch and a time dimension, got shape {tuple(inputs.shape)}")
        states: Tensor = inputs
        factors: Tensor = torch.broadcast_to(decays, inputs.shape)
        length: int = inputs.shape[1]
        shift: int = 1
        while shift < length:
            # After this step, each state sums the inputs of the last 2 * shift steps.
            states = torch.cat([states[:, :shift], factors[:, shift:] * states[:, :-shift] + states[:, shift:]], dim=1)
            factors = torch.cat([factors[:, :shift], factors[:, shift:] * factors[:, :-shift]], dim=1)
            shift *= 2
        return states
