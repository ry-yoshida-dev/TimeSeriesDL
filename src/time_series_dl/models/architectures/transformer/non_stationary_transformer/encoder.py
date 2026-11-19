from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from .de_stationary_factors import DeStationaryFactors
from .encoder_layer import NonStationaryEncoderLayer

if TYPE_CHECKING:
    from .parameters import NonStationaryTransformerParameters


class NonStationaryEncoder(nn.Module):
    """
    Value embedding, de-stationary encoder layers and a final layer normalization.

    Parameters
    ----------
    parameters : NonStationaryTransformerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: NonStationaryTransformerParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, parameters.model_size, parameters.dropout, input_length
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            NonStationaryEncoderLayer(
                parameters.model_size,
                parameters.num_heads,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.activation,
            )
            for _ in range(parameters.num_encoder_layers)
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)

    def forward(self, values: Tensor, factors: DeStationaryFactors) -> Tensor:
        """
        Encode a batch of windows.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, input_channels).
        factors : DeStationaryFactors
            The de-stationary factors of the windows.

        Returns
        -------
        Tensor
            Tokens with shape (batch, input_length, model_size).
        """
        tokens: Tensor = self.embedding(values)
        for layer in self.layers:
            tokens = cast(NonStationaryEncoderLayer, layer)(tokens, factors)
        normalized: Tensor = self.norm(tokens)
        return normalized

    def __call__(self, values: Tensor, factors: DeStationaryFactors) -> Tensor:
        return cast(Tensor, super().__call__(values, factors))
