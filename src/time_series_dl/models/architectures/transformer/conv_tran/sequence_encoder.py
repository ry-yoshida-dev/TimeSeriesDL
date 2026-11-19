from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.layers import Activation, MultilayerPerceptron
from ....interfaces import SequenceEncoder
from .attention import ConvTranAttention
from .convolutional_embedding import ConvTranConvolutionalEmbedding
from .position_encoding import ConvTranPositionEncoding

if TYPE_CHECKING:
    from .parameters import ConvTranParameters


class ConvTranSequenceEncoder(SequenceEncoder):
    """
    ConvTran encoder: convolutional embedding, one attention block whose queries, keys and values read the
    position-encoded tokens while the residual adds the plain tokens, and a ReLU feed-forward network, each followed by
    layer normalization.

    Parameters
    ----------
    parameters : ConvTranParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive.
    """

    def __init__(self, parameters: ConvTranParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.model_size
        self.embedding: ConvTranConvolutionalEmbedding = ConvTranConvolutionalEmbedding(
            input_channels, parameters.model_size, parameters.kernel_size
        )
        self.position: ConvTranPositionEncoding = ConvTranPositionEncoding(
            parameters.absolute_position_encoding, parameters.model_size, input_length, parameters.dropout
        )
        self.attention: ConvTranAttention = ConvTranAttention(
            parameters.relative_position_encoding,
            parameters.model_size,
            parameters.num_heads,
            input_length,
            parameters.dropout,
        )
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.feedforward: MultilayerPerceptron = MultilayerPerceptron(
            parameters.model_size,
            (parameters.feedforward_size,),
            parameters.model_size,
            Activation.RELU,
            parameters.dropout,
        )
        self.feedforward_dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.feedforward_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)

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
        attended: Tensor = self.attention_norm(tokens + self.attention(self.position(tokens)))
        transformed: Tensor = self.feedforward_dropout(self.feedforward(attended))
        outputs: Tensor = self.feedforward_norm(attended + transformed)
        return outputs
