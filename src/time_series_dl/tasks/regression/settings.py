from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...losses import MSELoss, PointLoss
from ...models import SummaryModel
from ...training import TrainingSettings

if TYPE_CHECKING:
    from .regressor import SequenceRegressor


@dataclass(frozen=True, kw_only=True)
class RegressorSettings:
    """
    Everything needed to train a sequence regressor, identical for every model.

    Attributes
    ----------
    model : SummaryModel
        Parameters of the network architecture, such as ``ResNetParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    loss : PointLoss
        Training loss on the scaled targets, such as ``HuberLoss(delta=0.5)`` or ``PinballLoss(0.9)``.
    scaling : ScalingParameters
        Per-channel scaling of the sequences, fitted on the training sequences.
    target_scaling : ScalingParameters
        Per-output scaling of the targets, fitted on the training targets; it must be invertible.
    """

    model: SummaryModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    loss: PointLoss = field(default_factory=MSELoss)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    target_scaling: ScalingParameters = field(default_factory=StandardScalingParameters)

    def build(self) -> SequenceRegressor:
        """
        Build an unfitted regressor.

        Returns
        -------
        SequenceRegressor
            The regressor.

        Raises
        ------
        ValueError
            If a scaling strategy mixes channels, or the target scaling is not invertible.
        """
        from .regressor import SequenceRegressor

        return SequenceRegressor(self)
