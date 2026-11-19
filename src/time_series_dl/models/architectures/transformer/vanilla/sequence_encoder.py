from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import TransformerParameters


class TransformerSequenceEncoder(SequenceEncoder):
    """
    Value embedding with sinusoidal position encoding followed by Transformer encoder layers.

    Parameters
    ----------
    parameters : TransformerParameters
        The layer configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: TransformerParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.model_size
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, parameters.model_size, parameters.dropout, input_length
        )
        layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            parameters.model_size,
            parameters.num_heads,
            parameters.feedforward_size,
            parameters.dropout,
            activation=parameters.activation.build(),
            batch_first=True,
            norm_first=parameters.is_pre_norm,
        )
        self.layers: nn.TransformerEncoder = nn.TransformerEncoder(
            layer,
            parameters.num_layers,
            norm=nn.LayerNorm(parameters.model_size) if parameters.is_pre_norm else None,
            enable_nested_tensor=False,
        )

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.layers(self.embedding(values))
        return features
