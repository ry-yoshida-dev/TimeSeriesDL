from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class ETSformerExponentialSmoothing(nn.Module):
    """
    Exponential smoothing attention (ESA) of ETSformer: each head smooths its values with a learnable rate.

    With the smoothing weight ``alpha = sigmoid(w)`` of a head, the output is ``s_t = alpha * s_{t-1} + (1 - alpha) *
    x_t`` from the learnable initial state ``s_{-1} = v_0``; with auxiliary values ``a_t`` it is ``s_t = alpha *
    (s_{t-1} + a_t) + (1 - alpha) * x_t``. The recurrence is evaluated in parallel as a product with the
    lower-triangular matrix of the attention weights ``(1 - alpha) * alpha^(t - j)``.

    Parameters
    ----------
    num_heads : int
        Number of heads, each with its own smoothing weight and initial state.
    head_size : int
        Size of the values of each head.
    dropout : float
        Dropout probability of the values and of the auxiliary values.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(self, num_heads: int, head_size: int, dropout: float) -> None:
        if num_heads <= 0 or head_size <= 0:
            raise ValueError(f"num_heads and head_size must be positive, got {num_heads} and {head_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.smoothing_logit: nn.Parameter = nn.Parameter(torch.randn(num_heads))
        self.initial_state: nn.Parameter = nn.Parameter(torch.randn(num_heads, head_size))
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.auxiliary_dropout: nn.Dropout = nn.Dropout(dropout)

    @property
    def smoothing_weight(self) -> Tensor:
        """
        Get the smoothing weight of each head, the weight of the previous state.

        Returns
        -------
        Tensor
            Weights in (0, 1) with shape (num_heads,).
        """
        return torch.sigmoid(self.smoothing_logit)

    def forward(self, values: Tensor, auxiliary_values: Tensor | None = None) -> Tensor:
        """
        Smooth a batch of value sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, num_heads, head_size).
        auxiliary_values : Tensor | None
            Values of the same shape added to the previous state before it is weighted, or None.

        Returns
        -------
        Tensor
            Smoothed states with shape (batch, length, num_heads, head_size).
        """
        steps: Tensor = torch.arange(values.shape[1], device=values.device)
        lags: Tensor = steps.unsqueeze(1) - steps.unsqueeze(0)
        log_weight: Tensor = functional.logsigmoid(self.smoothing_logit).view(-1, 1, 1)
        exponents: Tensor = (lags.clamp(min=0).to(values.dtype) * log_weight).masked_fill(lags < 0, float("-inf"))
        decay: Tensor = torch.exp(exponents)
        complement: Tensor = torch.sigmoid(-self.smoothing_logit).view(-1, 1, 1)
        smoothed: Tensor = torch.einsum("htj,bjhd->bthd", complement * decay, self.dropout(values))
        initial_weight: Tensor = torch.exp((steps + 1).to(values.dtype).unsqueeze(1) * log_weight.view(1, -1))
        outputs: Tensor = smoothed + initial_weight.unsqueeze(-1) * self.initial_state
        if auxiliary_values is not None:
            auxiliary_weight: Tensor = self.smoothing_weight.view(-1, 1, 1) * decay
            outputs = outputs + torch.einsum(
                "htj,bjhd->bthd", auxiliary_weight, self.auxiliary_dropout(auxiliary_values)
            )
        return outputs

    def __call__(self, values: Tensor, auxiliary_values: Tensor | None = None) -> Tensor:
        return cast(Tensor, super().__call__(values, auxiliary_values))
