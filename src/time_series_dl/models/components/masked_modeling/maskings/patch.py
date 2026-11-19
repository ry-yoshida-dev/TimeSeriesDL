import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..masking import Masking


@dataclass(frozen=True, kw_only=True)
class PatchMasking(Masking):
    """
    Masks of each channel that hide whole non-overlapping patches, as the self-supervised PatchTST (Nie et al., 2023).

    The sequence is cut into ``ceil(length / patch_length)`` patches, the last one possibly shorter, and exactly
    ``num_patches - int(num_patches * (1 - ratio))`` patches of every channel are hidden.

    Attributes
    ----------
    ratio : float
        Fraction of hidden patches.
    patch_length : int
        Number of time steps of each patch.

    Raises
    ------
    ValueError
        If ratio is not in (0, 1) or patch_length is not positive.
    """

    ratio: float = 0.4
    patch_length: int = 12

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {self.ratio}")
        if self.patch_length <= 0:
            raise ValueError(f"patch_length must be positive, got {self.patch_length}")

    def sample(self, batch: int, length: int, channels: int, device: torch.device) -> Tensor:
        num_patches: int = math.ceil(length / self.patch_length)
        num_visible: int = int(num_patches * (1.0 - self.ratio))
        ranks: Tensor = torch.rand(batch, channels, num_patches, device=device).argsort(dim=-1).argsort(dim=-1)
        is_patch_hidden: Tensor = ranks >= num_visible
        is_hidden: Tensor = is_patch_hidden.repeat_interleave(self.patch_length, dim=-1)[..., :length]
        return is_hidden.transpose(1, 2)
