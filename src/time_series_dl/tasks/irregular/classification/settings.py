from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ....models.interfaces.protocols.irregular_summary import IrregularSummaryModel
from ....training import TrainingSettings
from ...classification import ClassificationLoss, CrossEntropyLoss

if TYPE_CHECKING:
    from .classifier import IrregularSequenceClassifier


@dataclass(frozen=True, kw_only=True)
class IrregularClassifierSettings:
    """
    Everything needed to train a classifier of irregularly sampled sequences, identical for every model.

    Attributes
    ----------
    model : IrregularSummaryModel
        Parameters of the network architecture, such as ``MTANParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the observed training values.
    loss : ClassificationLoss
        Training loss, such as ``CrossEntropyLoss(label_smoothing=0.1)`` or ``FocalLoss(gamma=2.0)``.
    is_class_balanced : bool
        Whether the loss weights each class by its inverse frequency in the training labels, as for the imbalanced
        mortality labels of PhysioNet.
    """

    model: IrregularSummaryModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    loss: ClassificationLoss = field(default_factory=CrossEntropyLoss)
    is_class_balanced: bool = False

    def build(self) -> IrregularSequenceClassifier:
        """
        Build an unfitted classifier.

        Returns
        -------
        IrregularSequenceClassifier
            The classifier.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels.
        """
        from .classifier import IrregularSequenceClassifier

        return IrregularSequenceClassifier(self)
