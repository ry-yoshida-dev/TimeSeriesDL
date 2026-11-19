from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...losses import MSELoss, PointLoss
from ...models import ReconstructionModel
from ...models.interfaces.protocols.masked_reconstruction import MaskedReconstructionModel
from ...training import TrainingSettings

if TYPE_CHECKING:
    from .imputer import Imputer


@dataclass(frozen=True, kw_only=True)
class ImputerSettings:
    """
    Everything needed to train an imputer, identical for every model.

    The network learns to reconstruct observed values that are hidden at random during training, as in the
    imputation benchmark of Time-Series-Library and the masked imputation task of SAITS. Models that build a masked
    reconstruction network (SAITS, BRITS, GRU-D) also read the observed mask and add their own training loss.

    Attributes
    ----------
    model : ReconstructionModel | MaskedReconstructionModel
        Parameters of the network architecture, such as ``TimesNetParameters`` or ``SAITSParameters``; a model that
        builds both networks is used through its masked reconstruction network.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    mask_ratio : float
        Fraction of the observed values hidden in each training batch; validation hides a fixed random set.
    loss : PointLoss
        Loss on the scaled hidden values: weighted by the network's ``hidden_value_loss_weight`` while training, and
        alone for validation.
    scaling : ScalingParameters
        Per-channel scaling fitted on the observed training values; it must be invertible. Missing values are
        allowed whatever its ``is_nan_allowed``.

    Raises
    ------
    ValueError
        If mask_ratio is not in (0, 1).
    """

    model: ReconstructionModel | MaskedReconstructionModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    mask_ratio: float = 0.25
    loss: PointLoss = field(default_factory=MSELoss)
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")

    def build(self) -> Imputer:
        """
        Build an unfitted imputer.

        Returns
        -------
        Imputer
            The imputer.

        Raises
        ------
        ValueError
            If the scaling strategy is not invertible or mixes channels.
        """
        from .imputer import Imputer

        return Imputer(self)
