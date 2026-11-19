from typing import cast

import torch
from torch import Tensor, nn

from .decoder_layer import TranADDecoderLayer


class TranADDecoder(nn.Module):
    """
    Reconstructs a window from the encoder memory: the window, repeated to the token size, is the decoder target, and
    a linear layer and a sigmoid map the decoded tokens back to the channels, scaled to [0, 1].

    Parameters
    ----------
    channels : int
        Number of channels of the window.
    feedforward_size : int
        Hidden size of the feed-forward networks.
    num_layers : int
        Number of decoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(self, channels: int, feedforward_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            TranADDecoderLayer(2 * channels, channels, feedforward_size, dropout) for _ in range(num_layers)
        )
        self.output: nn.Sequential = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())

    def forward(self, values: Tensor, memory: Tensor) -> Tensor:
        """
        Reconstruct a batch of windows.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).
        memory : Tensor
            Encoder memory with shape (batch, length, 2 * channels).

        Returns
        -------
        Tensor
            Reconstructed windows with the shape of ``values``.
        """
        target: Tensor = torch.cat([values, values], dim=2)
        for layer in self.layers:
            target = layer(target, memory)
        reconstructed: Tensor = self.output(target)
        return reconstructed

    def __call__(self, values: Tensor, memory: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values, memory))
