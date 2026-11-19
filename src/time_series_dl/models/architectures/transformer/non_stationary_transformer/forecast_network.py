from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from ....interfaces import ForecastNetwork, ForecastShape
from .de_stationary_factors import DeStationaryFactors
from .decoder_layer import NonStationaryDecoderLayer
from .encoder import NonStationaryEncoder
from .factor_learner import DeStationaryFactorLearner
from .stationarization import Stationarization

if TYPE_CHECKING:
    from .parameters import NonStationaryTransformerParameters


class NonStationaryForecastNetwork(ForecastNetwork):
    """
    Non-stationary Transformer encoder-decoder.

    The past window is stationarized, encoded with de-stationary attention, and decoded from its last
    ``label_length`` stationarized steps followed by zeros for the horizon in one forward pass. The level parameters
    of each target channel are then restored to the scale of the window; the other parameters, such as standard
    deviations before their constraint, stay in the stationarized scale.

    Parameters
    ----------
    parameters : NonStationaryTransformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If label_length exceeds the input length.
    """

    def __init__(self, parameters: NonStationaryTransformerParameters, shape: ForecastShape) -> None:
        if parameters.label_length > shape.input_length:
            raise ValueError(f"label_length {parameters.label_length} exceeds the input length {shape.input_length}")
        super().__init__()
        self.label_length: int = parameters.label_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.factor_learner: DeStationaryFactorLearner = DeStationaryFactorLearner(
            shape.input_length, shape.input_channels, parameters.projector_hidden_sizes
        )
        self.encoder: NonStationaryEncoder = NonStationaryEncoder(parameters, shape.input_length, shape.input_channels)
        self.decoder_embedding: ValueEmbedding = ValueEmbedding(
            shape.input_channels, parameters.model_size, parameters.dropout, parameters.label_length + shape.horizon
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            NonStationaryDecoderLayer(
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
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)
        is_level: Tensor = torch.zeros(shape.num_parameters, dtype=torch.bool)
        is_level[list(shape.level_parameters)] = True
        self.is_level_parameter: Tensor
        self.register_buffer("is_level_parameter", is_level, persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        stationarization: Stationarization = Stationarization.from_values(past)
        normalized: Tensor = stationarization.normalize(past)
        factors: DeStationaryFactors = self.factor_learner(past, stationarization)
        memory: Tensor = self.encoder(normalized, factors)
        placeholder: Tensor = normalized.new_zeros(past.shape[0], self.horizon, past.shape[2])
        tokens: Tensor = self.decoder_embedding(
            torch.cat([normalized[:, past.shape[1] - self.label_length :], placeholder], dim=1)
        )
        for layer in self.decoder_layers:
            tokens = cast(NonStationaryDecoderLayer, layer)(tokens, memory, factors)
        outputs: Tensor = self.projection(self.decoder_norm(tokens[:, -self.horizon :])).reshape(
            -1, self.horizon, self.output_channels, self.num_parameters
        )
        mean: Tensor = stationarization.mean.index_select(2, self.target_channels).unsqueeze(-1)
        deviation: Tensor = stationarization.standard_deviation.index_select(2, self.target_channels).unsqueeze(-1)
        return torch.where(self.is_level_parameter, outputs * deviation + mean, outputs)
