from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters
from deep_learning_loss import MSELoss, PointLoss

from ....models.interfaces.protocols.irregular_summary import IrregularSummaryModel
from ....training import TrainingSettings

if TYPE_CHECKING:
    from .regressor import IrregularSequenceRegressor


@dataclass(frozen=True, kw_only=True)
class IrregularRegressorSettings:
    """
    Everything needed to train a regressor of irregularly sampled sequences, identical for every model.

    Attributes
    ----------
    model : IrregularSummaryModel
        Parameters of the network architecture, such as ``NeuralCDEParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    loss : PointLoss
        Training loss on the scaled targets, such as ``HuberLoss(delta=0.5)``.
    scaling : ScalingParameters
        Per-channel scaling fitted on the observed training values.
    target_scaling : ScalingParameters
        Per-output scaling of the targets, fitted on the training targets; it must be invertible.
    """

    model: IrregularSummaryModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    loss: PointLoss = field(default_factory=MSELoss)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    target_scaling: ScalingParameters = field(default_factory=StandardScalingParameters)

    def build(self) -> IrregularSequenceRegressor:
        """
        Build an unfitted regressor.

        Returns
        -------
        IrregularSequenceRegressor
            The regressor.

        Raises
        ------
        ValueError
            If a scaling strategy mixes channels, or the target scaling is not invertible.
        """
        from .regressor import IrregularSequenceRegressor

        return IrregularSequenceRegressor(self)
