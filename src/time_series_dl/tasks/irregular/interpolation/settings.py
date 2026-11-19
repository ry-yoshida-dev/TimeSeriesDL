from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ....models.interfaces.protocols.irregular_interpolation import IrregularInterpolationModel
from ....training import TrainingSettings
from .context_selection import ContextSelection

if TYPE_CHECKING:
    from .interpolator import IrregularInterpolator


@dataclass(frozen=True, kw_only=True)
class IrregularInterpolatorSettings:
    """
    Everything needed to train an interpolator or forecaster of irregularly sampled sequences, identical for every
    model. The training loss is the network's own, such as the negative evidence lower bound of the variational
    models.

    Attributes
    ----------
    model : IrregularInterpolationModel
        Parameters of the network architecture, such as ``LatentODEParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    scaling : ScalingParameters
        Per-channel scaling fitted on the observed training values; it must be invertible.
    context_selection : ContextSelection
        How training splits each sequence into context and targets.
    context_ratio : float
        Fraction of the observed values, or of the time range, that training gives as context; validation uses a
        fixed random selection.

    Raises
    ------
    ValueError
        If context_ratio is not in (0, 1).
    """

    model: IrregularInterpolationModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    context_selection: ContextSelection = ContextSelection.RANDOM_OBSERVATIONS
    context_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.context_ratio < 1.0:
            raise ValueError(f"context_ratio must be in (0, 1), got {self.context_ratio}")

    def build(self) -> IrregularInterpolator:
        """
        Build an unfitted interpolator.

        Returns
        -------
        IrregularInterpolator
            The interpolator.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels or is not invertible.
        """
        from .interpolator import IrregularInterpolator

        return IrregularInterpolator(self)
