from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .normalization import TSMixerNormalization


class TSMixerBlock(nn.Module):
    """
    Time-mixing MLP shared across channels, then feature-mixing MLP shared across time steps, each normalized in front
    and wrapped in a residual connection.

    Parameters
    ----------
    length : int
        Number of time steps.
    channels : int
        Number of channels.
    hidden_size : int
        Size of the hidden layer of the feature-mixing MLP.
    dropout : float
        Dropout probability after each mixing layer.
    normalization : TSMixerNormalization
        Normalization in front of each MLP.
    """

    def __init__(
        self, length: int, channels: int, hidden_size: int, dropout: float, normalization: TSMixerNormalization
    ) -> None:
        super().__init__()
        self.time_normalization: nn.Module = normalization.build(length, channels)
        self.time_linear: nn.Linear = nn.Linear(length, length)
        self.feature_normalization: nn.Module = normalization.build(length, channels)
        self.feature_hidden: nn.Linear = nn.Linear(channels, hidden_size)
        self.feature_output: nn.Linear = nn.Linear(hidden_size, channels)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Mix a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Mixed values with the same shape.
        """
        normalized: Tensor = self.time_normalization(values)
        mixed: Tensor = self.time_linear(normalized.transpose(1, 2)).transpose(1, 2)
        values = values + self.dropout(functional.relu(mixed))
        normalized = self.feature_normalization(values)
        hidden: Tensor = self.dropout(functional.relu(self.feature_hidden(normalized)))
        mixed = self.dropout(self.feature_output(hidden))
        return values + mixed

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
