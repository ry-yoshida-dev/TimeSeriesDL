from typing import override

import torch
from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from ....components.layers.series_decomposition import SeriesDecomposition
from ....interfaces import ForecastNetwork, ForecastShape
from .configuration import DecompositionTransformerConfiguration
from .decoder_layer import DecompositionDecoderLayer
from .seasonal_layer_norm import SeasonalLayerNorm
from .sequence_encoder import DecompositionTransformerSequenceEncoder


class DecompositionTransformerForecastNetwork(ForecastNetwork):
    """
    Encoder-decoder of Autoformer and FEDformer with progressive series decomposition.

    The past is decomposed into a seasonal part and a trend. The decoder reads the seasonal part of the last
    ``label_length`` steps followed by zeros for the horizon. Its trend starts from the trend of the same steps
    followed by the mean of the past, and accumulates the trend increments of every decoder layer. The forecast is the
    projected seasonal output plus, for the parameters in the scale of the values, the accumulated trend.

    Parameters
    ----------
    configuration : DecompositionTransformerConfiguration
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If label_length exceeds the input length.
    """

    def __init__(self, configuration: DecompositionTransformerConfiguration, shape: ForecastShape) -> None:
        if configuration.label_length > shape.input_length:
            raise ValueError(f"label_length {configuration.label_length} exceeds the input length {shape.input_length}")
        super().__init__()
        self.label_length: int = configuration.label_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        decoder_length: int = configuration.label_length + shape.horizon
        self.decomposition: SeriesDecomposition = SeriesDecomposition(configuration.moving_average_size)
        self.encoder: DecompositionTransformerSequenceEncoder = DecompositionTransformerSequenceEncoder(
            configuration, shape.input_length, shape.input_channels
        )
        self.decoder_embedding: ValueEmbedding = ValueEmbedding(
            shape.input_channels, configuration.model_size, configuration.dropout, decoder_length
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            DecompositionDecoderLayer(
                configuration.build_self_attention(decoder_length),
                configuration.build_cross_attention(decoder_length, shape.input_length),
                configuration.model_size,
                configuration.num_heads,
                configuration.feedforward_size,
                configuration.moving_average_size,
                configuration.dropout,
                configuration.activation,
                shape.output_channels,
            )
            for _ in range(configuration.num_decoder_layers)
        )
        self.decoder_norm: SeasonalLayerNorm = SeasonalLayerNorm(configuration.model_size)
        self.projection: nn.Linear = nn.Linear(configuration.model_size, shape.output_channels * shape.num_parameters)
        is_level: Tensor = torch.zeros(shape.num_parameters)
        is_level[list(shape.level_parameters)] = 1.0
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        start: int = past.shape[1] - self.label_length
        seasonal: Tensor
        trend: Tensor
        seasonal, trend = self.decomposition(past)
        target_mean: Tensor = past.index_select(2, self.target_channels).mean(dim=1, keepdim=True)
        accumulated_trend: Tensor = torch.cat(
            [trend[:, start:].index_select(2, self.target_channels), target_mean.expand(-1, self.horizon, -1)], 1
        )
        placeholder: Tensor = past.new_zeros(batch_size, self.horizon, past.shape[2])
        tokens: Tensor = self.decoder_embedding(torch.cat([seasonal[:, start:], placeholder], 1))
        memory: Tensor = self.encoder(past)
        for layer in self.decoder_layers:
            increment: Tensor
            tokens, increment = layer(tokens, memory)
            accumulated_trend = accumulated_trend + increment
        seasonal_output: Tensor = self.projection(self.decoder_norm(tokens)[:, -self.horizon :])
        outputs: Tensor = seasonal_output.reshape(batch_size, self.horizon, self.output_channels, self.num_parameters)
        return outputs + accumulated_trend[:, -self.horizon :].unsqueeze(-1) * self.is_level
