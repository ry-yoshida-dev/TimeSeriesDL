import math
from typing import cast

import torch
from torch import Tensor, nn

from ....components.layers import SinusoidalPositionalEncoding
from .encoder_layer import TranADEncoderLayer


class TranADEncoder(nn.Module):
    """
    Encodes a window concatenated with its focus score: scaled by the square root of the number of channels, position
    encoded and passed through the encoder layers.

    Parameters
    ----------
    channels : int
        Number of channels of the window; tokens have twice as many features and there is one head per channel.
    length : int
        Number of steps of the window.
    feedforward_size : int
        Hidden size of the feed-forward networks.
    num_layers : int
        Number of encoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(self, channels: int, length: int, feedforward_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.scale: float = math.sqrt(channels)
        self.position: SinusoidalPositionalEncoding = SinusoidalPositionalEncoding(2 * channels, length)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.layers: nn.Sequential = nn.Sequential(
            *(TranADEncoderLayer(2 * channels, channels, feedforward_size, dropout) for _ in range(num_layers))
        )

    def forward(self, values: Tensor, focus: Tensor) -> Tensor:
        """
        Encode a batch of windows and focus scores.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).
        focus : Tensor
            Focus scores with the same shape; zero in the first phase.

        Returns
        -------
        Tensor
            The memory with shape (batch, length, 2 * channels).
        """
        tokens: Tensor = self.dropout(self.position(torch.cat([values, focus], dim=2) * self.scale))
        memory: Tensor = self.layers(tokens)
        return memory

    def __call__(self, values: Tensor, focus: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values, focus))
