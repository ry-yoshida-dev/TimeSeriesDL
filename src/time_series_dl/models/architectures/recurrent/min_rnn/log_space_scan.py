import torch
from torch import Tensor


class LogSpaceScan:
    """
    Parallel evaluation in log space of the linear recurrence h_t = a_t * h_{t-1} + b_t from h_0 = 0, for positive
    a_t and b_t (Heinsen, 2023), as used by minGRU and minLSTM.

    With A_t = log a_1 + ... + log a_t, the states are h_t = exp(A_t + logcumsumexp_{k <= t}(log b_k - A_k)), a
    cumulative sum and a cumulative log-sum-exp along time, which is numerically stable for long sequences.
    """

    @staticmethod
    def scan(log_coefficients: Tensor, log_values: Tensor) -> Tensor:
        """
        Evaluate the recurrence along the second dimension.

        Parameters
        ----------
        log_coefficients : Tensor
            Logarithms of the factors a_t with shape (batch, length, ...).
        log_values : Tensor
            Logarithms of the inputs b_t with the shape of ``log_coefficients``.

        Returns
        -------
        Tensor
            States h_t with the shape of the inputs.

        Raises
        ------
        ValueError
            If the shapes differ or have no time dimension.
        """
        if log_coefficients.shape != log_values.shape:
            raise ValueError(f"shapes must match, got {tuple(log_coefficients.shape)} and {tuple(log_values.shape)}")
        if log_values.dim() < 2:
            raise ValueError(f"inputs must have a batch and a time dimension, got shape {tuple(log_values.shape)}")
        cumulative: Tensor = log_coefficients.cumsum(dim=1)
        return torch.exp(cumulative + torch.logcumsumexp(log_values - cumulative, dim=1))
