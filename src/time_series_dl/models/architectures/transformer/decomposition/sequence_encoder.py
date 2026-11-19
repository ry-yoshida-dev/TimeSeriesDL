from typing import override

from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from ....interfaces import SequenceEncoder
from .configuration import DecompositionTransformerConfiguration
from .encoder_layer import DecompositionEncoderLayer
from .seasonal_layer_norm import SeasonalLayerNorm


class DecompositionTransformerSequenceEncoder(SequenceEncoder):
    """
    Encoder of a decomposition Transformer: value embedding with position encoding, decomposition encoder layers
    and a seasonal layer normalization.

    Parameters
    ----------
    configuration : DecompositionTransformerConfiguration
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(
        self, configuration: DecompositionTransformerConfiguration, input_length: int, input_channels: int
    ) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = configuration.model_size
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, configuration.model_size, configuration.dropout, input_length
        )
        self.layers: nn.Sequential = nn.Sequential(
            *(
                DecompositionEncoderLayer(
                    configuration.build_self_attention(input_length),
                    configuration.model_size,
                    configuration.num_heads,
                    configuration.feedforward_size,
                    configuration.moving_average_size,
                    configuration.dropout,
                    configuration.activation,
                )
                for _ in range(configuration.num_encoder_layers)
            )
        )
        self.norm: SeasonalLayerNorm = SeasonalLayerNorm(configuration.model_size)

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
        tokens: Tensor = self.layers(self.embedding(values))
        return self.norm(tokens)
