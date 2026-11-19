from typing import cast

import torch
from torch import Tensor, nn


class ScalarEmbedding(nn.Module):
    """
    Linear embedding of each scalar channel into a vector, with its own weights per channel, as TFT embeds real-valued
    inputs.

    Parameters
    ----------
    channels : int
        Number of channels.
    model_size : int
        Size of each embedding.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, channels: int, model_size: int) -> None:
        if channels <= 0 or model_size <= 0:
            raise ValueError(f"channels and model_size must be positive, got {channels} and {model_size}")
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.randn(channels, model_size))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(channels, model_size))

    def forward(self, values: Tensor) -> Tensor:
        """
        Embed every channel.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., channels).

        Returns
        -------
        Tensor
            Embeddings with shape (..., channels, model_size).
        """
        return values.unsqueeze(-1) * self.weight + self.bias

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
