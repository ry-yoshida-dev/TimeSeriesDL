from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class MaskedReconstruction:
    """
    Output of a masked reconstruction network for one batch.

    Attributes
    ----------
    values : Tensor
        Reconstruction of every value with shape (batch, length, channels), observed or not.
    loss : Tensor | None
        Scalar training loss that the network defines on its own inputs, such as the loss on the observed values
        of SAITS or the consistency loss of BRITS; None for networks trained only on the values hidden by the task.

    Raises
    ------
    ValueError
        If values does not have three dimensions or loss is not a scalar.
    """

    values: Tensor
    loss: Tensor | None

    def __post_init__(self) -> None:
        if self.values.dim() != 3:
            raise ValueError(f"values must have shape (batch, length, channels), got {tuple(self.values.shape)}")
        if self.loss is not None and self.loss.dim() != 0:
            raise ValueError(f"loss must be a scalar, got shape {tuple(self.loss.shape)}")
