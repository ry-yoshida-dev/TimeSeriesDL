from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.layers import SinusoidalPositionalEncoding
from .encoder_layer import SAITSEncoderLayer

if TYPE_CHECKING:
    from .parameters import SAITSParameters


class SAITSBlock(nn.Module):
    """
    Diagonally masked self-attention block: the values and the observed mask are embedded together, given position
    encodings, and passed through the encoder layers.

    Parameters
    ----------
    parameters : SAITSParameters
        The architecture.
    length : int
        Number of time steps.
    channels : int
        Number of channels.
    """

    def __init__(self, parameters: SAITSParameters, length: int, channels: int) -> None:
        super().__init__()
        self.embedding: nn.Linear = nn.Linear(2 * channels, parameters.model_size)
        self.positional_encoding: SinusoidalPositionalEncoding = SinusoidalPositionalEncoding(
            parameters.model_size, length
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.ModuleList = nn.ModuleList(
            SAITSEncoderLayer(
                parameters.model_size,
                parameters.num_heads,
                parameters.key_size,
                parameters.value_size,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.attention_dropout,
            )
            for _ in range(parameters.num_layers)
        )

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode a batch.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).
        mask : Tensor
            Float observed mask of the shape of ``values``.

        Returns
        -------
        tuple[Tensor, Tensor]
            The tokens with shape (batch, length, model_size), and the attention weights of the last layer averaged
            over the heads, with shape (batch, length, length).
        """
        tokens: Tensor = self.dropout(self.positional_encoding(self.embedding(torch.cat([values, mask], dim=2))))
        weights: Tensor = tokens.new_zeros(0)
        for layer in self.layers:
            tokens, weights = cast(SAITSEncoderLayer, layer)(tokens)
        return tokens, weights.mean(dim=1)

    def __call__(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values, mask))
