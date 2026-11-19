from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .de_stationary_factors import DeStationaryFactors
from .encoder import NonStationaryEncoder
from .factor_learner import DeStationaryFactorLearner
from .stationarization import Stationarization

if TYPE_CHECKING:
    from .parameters import NonStationaryTransformerParameters


class NonStationaryReconstructionNetwork(ReconstructionNetwork):
    """
    Stationarized sequence encoded with de-stationary attention, each step's tokens mapped linearly to its values and
    restored to the scale of the sequence.

    Parameters
    ----------
    parameters : NonStationaryTransformerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: NonStationaryTransformerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.factor_learner: DeStationaryFactorLearner = DeStationaryFactorLearner(
            shape.length, shape.channels, parameters.projector_hidden_sizes
        )
        self.encoder: NonStationaryEncoder = NonStationaryEncoder(parameters, shape.length, shape.channels)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.channels)

    @override
    def forward(self, values: Tensor) -> Tensor:
        stationarization: Stationarization = Stationarization.from_values(values)
        factors: DeStationaryFactors = self.factor_learner(values, stationarization)
        tokens: Tensor = self.encoder(stationarization.normalize(values), factors)
        return stationarization.restore(self.projection(tokens))
