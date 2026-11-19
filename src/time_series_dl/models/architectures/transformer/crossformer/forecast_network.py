from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .decoder_layer import CrossformerDecoderLayer
from .encoder import CrossformerEncoder

if TYPE_CHECKING:
    from .parameters import CrossformerParameters


class CrossformerForecastNetwork(ForecastNetwork):
    """
    Crossformer encoder-decoder: learnable position tokens of the output segments pass through one decoder layer per
    encoder scale, each layer predicts the segments linearly, and the forecast is the sum of the layers' predictions.

    Every channel is decoded, as the cross-dimension stage mixes them, but only the target channels are predicted.

    Parameters
    ----------
    parameters : CrossformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: CrossformerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: CrossformerEncoder = CrossformerEncoder(parameters, shape.input_length, shape.input_channels)
        output_segments: int = parameters.count_segments(shape.horizon)
        self.decoder_position: nn.Parameter = nn.Parameter(
            torch.randn(shape.input_channels, output_segments, parameters.model_size)
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            CrossformerDecoderLayer(
                output_segments,
                parameters.num_routers,
                parameters.model_size,
                parameters.num_heads,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.activation,
            )
            for _ in range(parameters.num_layers + 1)
        )
        self.predictions: nn.ModuleList = nn.ModuleList(
            nn.Linear(parameters.model_size, parameters.segment_length * shape.num_parameters)
            for _ in range(parameters.num_layers + 1)
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        scales: list[Tensor] = self.encoder(past)
        tokens: Tensor = self.decoder_position.expand(past.shape[0], -1, -1, -1)
        layer_predictions: list[Tensor] = []
        for layer, prediction, memory in zip(self.decoder_layers, self.predictions, scales, strict=True):
            tokens = layer(tokens, memory)
            layer_predictions.append(prediction(tokens.index_select(1, self.target_channels)))
        summed: Tensor = torch.stack(layer_predictions).sum(dim=0)
        series: Tensor = summed.reshape(past.shape[0], summed.shape[1], -1, self.num_parameters)
        return series[:, :, : self.horizon].transpose(1, 2)
