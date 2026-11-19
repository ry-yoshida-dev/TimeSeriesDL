from typing import cast

import torch
from torch import Tensor, nn

from .exponential_smoothing import ETSformerExponentialSmoothing


class ETSformerGrowthLayer(nn.Module):
    """
    Growth of ETSformer: exponential smoothing attention over the successive differences of the projected tokens.

    The first difference is taken from a learnable initial value, and the initial state of the smoothing is prepended,
    so the output is one step longer than the input: position ``t`` holds the growth known before step ``t``.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of smoothing heads; it divides model_size.
    dropout : float
        Dropout probability of the smoothed differences.

    Raises
    ------
    ValueError
        If a size is not positive or num_heads does not divide model_size.
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float) -> None:
        if model_size <= 0 or num_heads <= 0 or model_size % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide model_size, got {num_heads} and {model_size}")
        super().__init__()
        self.num_heads: int = num_heads
        self.initial_value: nn.Parameter = nn.Parameter(torch.randn(num_heads, model_size // num_heads))
        self.input_projection: nn.Linear = nn.Linear(model_size, model_size)
        self.smoothing: ETSformerExponentialSmoothing = ETSformerExponentialSmoothing(
            num_heads, model_size // num_heads, dropout
        )
        self.output_projection: nn.Linear = nn.Linear(model_size, model_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Extract the growth of a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Growth representations with shape (batch, length + 1, model_size).
        """
        batch: int = tokens.shape[0]
        values: Tensor = self.input_projection(tokens).reshape(batch, tokens.shape[1], self.num_heads, -1)
        padded: Tensor = torch.cat([self.initial_value.expand(batch, 1, -1, -1), values], dim=1)
        smoothed: Tensor = self.smoothing(padded[:, 1:] - padded[:, :-1])
        initial: Tensor = self.smoothing.initial_state.expand(batch, 1, -1, -1)
        growth: Tensor = torch.cat([initial, smoothed], dim=1).flatten(start_dim=2)
        projected: Tensor = self.output_projection(growth)
        return projected

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
