from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.attention.attention_mechanisms.full import FullAttentionMechanism
from ....components.attention.attention_mechanisms.prob_sparse import ProbSparseAttentionMechanism
from ....components.attention.value_embedding import ValueEmbedding
from ....interfaces import ForecastNetwork, ForecastShape
from .decoder_layer import InformerDecoderLayer
from .sequence_encoder import InformerSequenceEncoder

if TYPE_CHECKING:
    from .parameters import InformerParameters


class InformerForecastNetwork(ForecastNetwork):
    """
    Informer encoder-decoder with a generative-style decoder.

    The decoder reads the last ``label_length`` past steps followed by zeros for the horizon, and outputs the whole
    horizon in one forward pass through masked ProbSparse self-attention and full cross-attention to the encoder.

    Parameters
    ----------
    parameters : InformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If label_length exceeds the input length.
    """

    def __init__(self, parameters: InformerParameters, shape: ForecastShape) -> None:
        if parameters.label_length > shape.input_length:
            raise ValueError(f"label_length {parameters.label_length} exceeds the input length {shape.input_length}")
        super().__init__()
        self.label_length: int = parameters.label_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.encoder: InformerSequenceEncoder = InformerSequenceEncoder(
            parameters, shape.input_length, shape.input_channels, parameters.is_distilling
        )
        self.decoder_embedding: ValueEmbedding = ValueEmbedding(
            shape.input_channels, parameters.model_size, parameters.dropout, parameters.label_length + shape.horizon
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            InformerDecoderLayer(
                ProbSparseAttentionMechanism(parameters.factor, True, parameters.dropout),
                FullAttentionMechanism(False, parameters.dropout),
                parameters.model_size,
                parameters.num_heads,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.activation,
            )
            for _ in range(parameters.num_decoder_layers)
        )
        self.decoder_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.output_channels * shape.num_parameters)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        memory: Tensor = self.encoder(past)
        placeholder: Tensor = past.new_zeros(past.shape[0], self.horizon, past.shape[2])
        tokens: Tensor = self.decoder_embedding(
            torch.cat([past[:, past.shape[1] - self.label_length :], placeholder], 1)
        )
        for layer in self.decoder_layers:
            tokens = layer(tokens, memory)
        outputs: Tensor = self.projection(self.decoder_norm(tokens[:, -self.horizon :]))
        return outputs.reshape(-1, self.horizon, self.output_channels, self.num_parameters)
