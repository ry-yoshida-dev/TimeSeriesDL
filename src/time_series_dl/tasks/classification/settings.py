from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...models import SummaryModel
from ...training import TrainingSettings
from .classification_loss import ClassificationLoss
from .classification_losses import CrossEntropyLoss

if TYPE_CHECKING:
    from .classifier import SequenceClassifier


@dataclass(frozen=True, kw_only=True)
class ClassifierSettings:
    """
    Everything needed to train a sequence classifier, identical for every model.

    Attributes
    ----------
    model : SummaryModel
        Parameters of the network architecture, such as ``InceptionTimeParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training sequences.
    loss : ClassificationLoss
        Training loss, such as ``CrossEntropyLoss(label_smoothing=0.1)`` or ``FocalLoss(gamma=2.0)``.
    is_class_balanced : bool
        Whether the loss weights each class by its inverse frequency in the training labels.
    """

    model: SummaryModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    loss: ClassificationLoss = field(default_factory=CrossEntropyLoss)
    is_class_balanced: bool = False

    def build(self) -> SequenceClassifier:
        """
        Build an unfitted classifier.

        Returns
        -------
        SequenceClassifier
            The classifier.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels.
        """
        from .classifier import SequenceClassifier

        return SequenceClassifier(self)
