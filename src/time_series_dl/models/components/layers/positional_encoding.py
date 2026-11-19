import math
from typing import cast

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds the fixed sinusoidal position encoding of the original Transformer.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    max_length : int
        Longest sequence that can be encoded.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, model_size: int, max_length: int) -> None:
        if model_size <= 0 or max_length <= 0:
            raise ValueError(f"model_size and max_length must be positive, got {model_size} and {max_length}")
        super().__init__()
        position: Tensor = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency: Tensor = torch.exp(
            torch.arange(0, model_size, 2, dtype=torch.float32) * (-math.log(10000.0) / model_size)
        )
        encoding: Tensor = torch.zeros(max_length, model_size)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: model_size // 2])
        self.encoding: Tensor
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Add the encoding of each position.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Tokens with their position encodings added.

        Raises
        ------
        ValueError
            If the sequence is longer than ``max_length``.
        """
        length: int = tokens.shape[1]
        if length > self.encoding.shape[0]:
            raise ValueError(f"sequence length {length} exceeds max_length {self.encoding.shape[0]}")
        return tokens + self.encoding[:length].unsqueeze(0)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
