from dataclasses import dataclass
from typing import override

from .....losses import PointLoss
from .....models import ForecastingModel, ForecastShape, ReconstructionModel, SequenceShape
from .....models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from ...scoring_step import AnomalyScoringStep
from ..anomaly_scoring import AnomalyScoring
from .anomaly_scoring_network import ForecastErrorScoringNetwork


@dataclass(frozen=True, kw_only=True)
class ForecastErrorScoring(AnomalyScoring):
    """
    Turns any forecasting model into a detector: a point forecast network trained on normal windows predicts the next
    ``horizon`` steps of every channel from the ``window_length`` steps before them, and each step is scored by its
    forecast loss, averaged over channels and over every forecast that covers it (one forecast per step with the
    default one-step-ahead horizon).

    The first ``window_length`` steps of a series have no forecast; they receive the lowest score of the series.

    Attributes
    ----------
    horizon : int
        Number of steps forecast from each past window.
    is_instance_normalized : bool
        Whether each past window is normalized by its own mean and standard deviation (RevIN), as in forecasting.

    Raises
    ------
    ValueError
        If horizon is not positive.
    """

    horizon: int = 1
    is_instance_normalized: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

    @override
    def is_supported(self, model: ReconstructionModel | ForecastingModel | AnomalyScoringModel) -> bool:
        return isinstance(model, ForecastingModel)

    @override
    def compute_sample_length(self, window_length: int) -> int:
        return window_length + self.horizon

    @override
    def build_step(
        self,
        model: ReconstructionModel | ForecastingModel | AnomalyScoringModel,
        shape: SequenceShape,
        loss: PointLoss,
    ) -> AnomalyScoringStep:
        if not isinstance(model, ForecastingModel):
            raise TypeError(f"{type(model).__name__} does not build a forecast network.")
        forecast_shape: ForecastShape = ForecastShape(
            input_length=shape.length,
            input_channels=shape.channels,
            horizon=self.horizon,
            target_channels=tuple(range(shape.channels)),
        )
        network = model.build_forecast_network(forecast_shape)
        return AnomalyScoringStep(
            network=ForecastErrorScoringNetwork(network, loss, forecast_shape, self.is_instance_normalized),
            model_network=network,
        )
