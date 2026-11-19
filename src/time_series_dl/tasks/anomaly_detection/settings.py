from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...losses import MSELoss, PointLoss
from ...models import ForecastingModel, ReconstructionModel
from ...models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from ...training import TrainingSettings
from .scoring_methods import AnomalyScoring, ForecastErrorScoring, NativeScoring, ReconstructionErrorScoring

if TYPE_CHECKING:
    from .detector import AnomalyDetector


@dataclass(frozen=True, kw_only=True)
class AnomalyDetectorSettings:
    """
    Everything needed to train an anomaly detector on normal behavior, identical for every model.

    Attributes
    ----------
    model : ReconstructionModel | ForecastingModel | AnomalyScoringModel
        Parameters of the network architecture: a reconstruction model such as ``RNNParameters`` (an LSTM
        autoencoder), any forecasting model, or an anomaly detection model with its own objective and score, such as
        ``AnomalyTransformerParameters``.
    window_length : int
        Number of time steps each window holds; for forecast-error scoring, the past steps each forecast reads.
    training_stride : int
        Steps between consecutive training windows; scoring always uses every window.
    threshold_quantile : float
        Quantile of the training scores above which a time step is anomalous.
    scoring : AnomalyScoring | None
        How the model is trained and scores steps: ``ReconstructionErrorScoring``, ``ForecastErrorScoring`` or
        ``NativeScoring``. When None, the model's own score if it builds an anomaly scoring network, otherwise its
        reconstruction error, otherwise its one-step-ahead forecast error.
    loss : PointLoss
        Loss of each scaled value for reconstruction- and forecast-error scoring, used both for training and for the
        anomaly scores; models with their own objective ignore it.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training series.

    Raises
    ------
    ValueError
        If a length is not positive or threshold_quantile is not in (0, 1].
    TypeError
        If the model does not build the network the scoring needs.
    """

    model: ReconstructionModel | ForecastingModel | AnomalyScoringModel
    window_length: int
    training_stride: int = 1
    threshold_quantile: float = 0.99
    scoring: AnomalyScoring | None = None
    loss: PointLoss = field(default_factory=MSELoss)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)

    def __post_init__(self) -> None:
        if self.window_length <= 0:
            raise ValueError(f"window_length must be positive, got {self.window_length}")
        if self.training_stride <= 0:
            raise ValueError(f"training_stride must be positive, got {self.training_stride}")
        if not 0.0 < self.threshold_quantile <= 1.0:
            raise ValueError(f"threshold_quantile must be in (0, 1], got {self.threshold_quantile}")
        if not self.resolved_scoring.is_supported(self.model):
            scoring_name: str = type(self.resolved_scoring).__name__
            raise TypeError(f"{type(self.model).__name__} does not support {scoring_name}.")

    @property
    def resolved_scoring(self) -> AnomalyScoring:
        """
        Get the scoring method in use.

        Returns
        -------
        AnomalyScoring
            ``scoring`` when set, otherwise the method chosen for the model.
        """
        if self.scoring is not None:
            return self.scoring
        if isinstance(self.model, AnomalyScoringModel):
            return NativeScoring()
        if isinstance(self.model, ReconstructionModel):
            return ReconstructionErrorScoring()
        return ForecastErrorScoring()

    def build(self) -> AnomalyDetector:
        """
        Build an unfitted anomaly detector.

        Returns
        -------
        AnomalyDetector
            The detector.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels.
        """
        from .detector import AnomalyDetector

        return AnomalyDetector(self)
