from dataclasses import dataclass
from typing import override

from deep_learning_loss import PointLoss

from .....models import ForecastingModel, ReconstructionModel, SequenceShape
from .....models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from ...scoring_step import AnomalyScoringStep
from ..anomaly_scoring import AnomalyScoring
from .anomaly_scoring_network import ReconstructionErrorScoringNetwork


@dataclass(frozen=True, kw_only=True)
class ReconstructionErrorScoring(AnomalyScoring):
    """
    Trains a reconstruction network on normal windows and scores each step by its reconstruction loss, averaged over
    channels and over every window that contains the step.
    """

    @override
    def is_supported(self, model: ReconstructionModel | ForecastingModel | AnomalyScoringModel) -> bool:
        return isinstance(model, ReconstructionModel)

    @override
    def compute_sample_length(self, window_length: int) -> int:
        return window_length

    @override
    def build_step(
        self,
        model: ReconstructionModel | ForecastingModel | AnomalyScoringModel,
        shape: SequenceShape,
        loss: PointLoss,
    ) -> AnomalyScoringStep:
        if not isinstance(model, ReconstructionModel):
            raise TypeError(f"{type(model).__name__} does not build a reconstruction network.")
        network = model.build_reconstruction_network(shape)
        return AnomalyScoringStep(network=ReconstructionErrorScoringNetwork(network, loss), model_network=network)
