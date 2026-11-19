from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.attention.attention_mechanisms.prob_sparse import ProbSparseAttentionMechanism
from ....components.attention.value_embedding import ValueEmbedding
from ....interfaces import SequenceEncoder
from .distilling_layer import InformerDistillingLayer
from .encoder_layer import InformerEncoderLayer

if TYPE_CHECKING:
    from .parameters import InformerParameters


class InformerSequenceEncoder(SequenceEncoder):
    """
    Informer encoder: value embedding with position encoding, ProbSparse self-attention layers, optionally a
    distilling layer between consecutive layers, and a final layer normalization.

    Parameters
    ----------
    parameters : InformerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    is_distilling : bool
        Whether a distilling layer halves the length between consecutive attention layers.
    """

    def __init__(
        self, parameters: InformerParameters, input_length: int, input_channels: int, is_distilling: bool
    ) -> None:
        super().__init__()
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, parameters.model_size, parameters.dropout, input_length
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            InformerEncoderLayer(
                ProbSparseAttentionMechanism(parameters.factor, False, parameters.dropout),
                parameters.model_size,
                parameters.num_heads,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.activation,
            )
            for _ in range(parameters.num_encoder_layers)
        )
        num_distilling: int = parameters.num_encoder_layers - 1 if is_distilling else 0
        self.distilling: nn.ModuleList = nn.ModuleList(
            InformerDistillingLayer(parameters.model_size) for _ in range(num_distilling)
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        output_length: int = input_length
        for _ in range(num_distilling):
            output_length = InformerDistillingLayer.output_length(output_length)
        self._output_length: int = output_length
        self._output_size: int = parameters.model_size

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
        tokens: Tensor = self.embedding(values)
        for position, layer in enumerate(self.layers):
            tokens = layer(tokens)
            if position < len(self.distilling):
                tokens = self.distilling[position](tokens)
        features: Tensor = self.norm(tokens)
        return features
