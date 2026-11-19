from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...losses import MSELoss, PointLoss
from ...models import ReconstructionModel
from ...training import TrainingSettings

if TYPE_CHECKING:
    from .detector import AnomalyDetector


@dataclass(frozen=True, kw_only=True)
class AnomalyDetectorSettings:
    """
    Everything needed to train a reconstruction-based anomaly detector, identical for every model.

    Attributes
    ----------
    model : ReconstructionModel
        Parameters of the network architecture, such as ``RNNParameters`` (an LSTM autoencoder).
    window_length : int
        Number of time steps the network reconstructs at once.
    training_stride : int
        Steps between consecutive training windows; scoring always uses every window.
    threshold_quantile : float
        Quantile of the training scores above which a time step is anomalous.
    loss : PointLoss
        Reconstruction loss of each scaled value, used both for training and for the anomaly scores.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training series.

    Raises
    ------
    ValueError
        If a length is not positive or threshold_quantile is not in (0, 1].
    """

    model: ReconstructionModel
    window_length: int
    training_stride: int = 1
    threshold_quantile: float = 0.99
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
