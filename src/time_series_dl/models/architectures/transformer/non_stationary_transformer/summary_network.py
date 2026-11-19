from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .de_stationary_factors import DeStationaryFactors
from .encoder import NonStationaryEncoder
from .factor_learner import DeStationaryFactorLearner
from .stationarization import Stationarization

if TYPE_CHECKING:
    from .parameters import NonStationaryTransformerParameters


class NonStationarySummaryNetwork(SummaryNetwork):
    """
    Raw sequence encoded with de-stationary attention whose factors are learned from its statistics, followed by an
    activation, dropout and a linear map of all flattened tokens to the output vector.

    As in Time-Series-Library, the sequence is not stationarized before the embedding, so its level and scale stay
    available to the output.

    Parameters
    ----------
    parameters : NonStationaryTransformerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: NonStationaryTransformerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.factor_learner: DeStationaryFactorLearner = DeStationaryFactorLearner(
            shape.input_length, shape.input_channels, parameters.projector_hidden_sizes
        )
        self.encoder: NonStationaryEncoder = NonStationaryEncoder(parameters, shape.input_length, shape.input_channels)
        self.activation: nn.Module = parameters.activation.build()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size * shape.input_length, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        factors: DeStationaryFactors = self.factor_learner(values, Stationarization.from_values(values))
        tokens: Tensor = self.activation(self.encoder(values, factors))
        outputs: Tensor = self.projection(self.dropout(tokens).flatten(start_dim=1))
        return outputs
