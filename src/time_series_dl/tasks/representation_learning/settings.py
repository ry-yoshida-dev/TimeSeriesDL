from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...models.interfaces.protocols.pretraining import PretrainingModel
from ...training import TrainingSettings

if TYPE_CHECKING:
    from .learner import RepresentationLearner


@dataclass(frozen=True, kw_only=True)
class RepresentationLearnerSettings:
    """
    Everything needed to pretrain an encoder on unlabeled sequences, identical for every method.

    Attributes
    ----------
    method : PretrainingModel
        Parameters of the pretraining method and its encoder, such as ``TS2VecParameters`` or
        ``SimMTMParameters(encoder=TSTParameters())``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training sequences.
    """

    method: PretrainingModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)

    def build(self) -> RepresentationLearner:
        """
        Build an unfitted representation learner.

        Returns
        -------
        RepresentationLearner
            The learner.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels.
        """
        from .learner import RepresentationLearner

        return RepresentationLearner(self)
