from typing import cast

import torch
from torch import Tensor, nn


class ETSformerDampingLayer(nn.Module):
    """
    Damped trend of ETSformer: the last growth repeated over the horizon, step ``h`` scaled by
    ``phi + phi^2 + ... + phi^h`` with a learnable damping factor ``phi = sigmoid(w)`` per head.

    Parameters
    ----------
    horizon : int
        Number of forecast steps.
    num_heads : int
        Number of heads, each with its own damping factor; it divides the size of the growth.
    dropout : float
        Dropout probability of the repeated growth.

    Raises
    ------
    ValueError
        If horizon or num_heads is not positive.
    """

    def __init__(self, horizon: int, num_heads: int, dropout: float) -> None:
        if horizon <= 0 or num_heads <= 0:
            raise ValueError(f"horizon and num_heads must be positive, got {horizon} and {num_heads}")
        super().__init__()
        self.horizon: int = horizon
        self.num_heads: int = num_heads
        self.damping_logit: nn.Parameter = nn.Parameter(torch.randn(num_heads))
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @property
    def damping_factor(self) -> Tensor:
        """
        Get the damping factor of each head.

        Returns
        -------
        Tensor
            Factors in (0, 1) with shape (num_heads,).
        """
        return torch.sigmoid(self.damping_logit)

    def forward(self, growth: Tensor) -> Tensor:
        """
        Extrapolate the last growth over the horizon.

        Parameters
        ----------
        growth : Tensor
            Last growth with shape (batch, 1, model_size).

        Returns
        -------
        Tensor
            Damped growth with shape (batch, horizon, model_size).
        """
        batch: int = growth.shape[0]
        powers: Tensor = torch.arange(1, self.horizon + 1, device=growth.device, dtype=growth.dtype).unsqueeze(1)
        factors: Tensor = (self.damping_factor**powers).cumsum(dim=0)
        repeated: Tensor = self.dropout(growth.expand(batch, self.horizon, -1))
        damped: Tensor = repeated.reshape(batch, self.horizon, self.num_heads, -1) * factors.unsqueeze(-1)
        return damped.flatten(start_dim=2)

    def __call__(self, growth: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(growth))
