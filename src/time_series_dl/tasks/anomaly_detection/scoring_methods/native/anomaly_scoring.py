from dataclasses import dataclass
from typing import override

from .....losses import PointLoss
from .....models import ForecastingModel, ReconstructionModel, SequenceShape
from .....models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from ...scoring_step import AnomalyScoringStep
from ..anomaly_scoring import AnomalyScoring


@dataclass(frozen=True, kw_only=True)
class NativeScoring(AnomalyScoring):
    """
    Trains an anomaly scoring network with the model's own objective and scores each step by the model's own anomaly
    score, averaged over every window that contains the step, such as the association-based criterion of the Anomaly
    Transformer.
    """

    @override
    def is_supported(self, model: ReconstructionModel | ForecastingModel | AnomalyScoringModel) -> bool:
        return isinstance(model, AnomalyScoringModel)

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
        if not isinstance(model, AnomalyScoringModel):
            raise TypeError(f"{type(model).__name__} does not build an anomaly scoring network.")
        network = model.build_anomaly_scoring_network(shape)
        return AnomalyScoringStep(network=network, model_network=network)
