from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class InstanceStatistics:
    """
    Location and scale of each window and channel, kept to undo a reversible instance normalization.

    Attributes
    ----------
    location : Tensor
        The mean with shape (batch, 1, channels).
    scale : Tensor
        The standard deviation with shape (batch, 1, channels).
    """

    location: Tensor
    scale: Tensor
