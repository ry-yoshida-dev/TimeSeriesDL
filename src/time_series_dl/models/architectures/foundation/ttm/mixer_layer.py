from typing import cast

from torch import Tensor, nn

from .feature_mixer import TTMFeatureMixer
from .patch_mixer import TTMPatchMixer


class TTMMixerLayer(nn.Module):
    """
    TinyTimeMixer layer: patch mixing (when there is more than one patch) and then feature mixing, each channel on its
    own.

    Parameters
    ----------
    num_patches : int
        Number of patches.
    size : int
        Number of features per patch.
    expansion_factor : int
        Ratio of the perceptrons' hidden sizes to their input sizes.
    dropout : float
        Dropout probability.
    is_gated : bool
        Whether the gated attention follows each perceptron.
    epsilon : float
        Epsilon of the layer normalizations.
    """

    def __init__(
        self, num_patches: int, size: int, expansion_factor: int, dropout: float, is_gated: bool, epsilon: float
    ) -> None:
        super().__init__()
        self.patch_mixer: TTMPatchMixer | None = (
            TTMPatchMixer(num_patches, size, expansion_factor, dropout, is_gated, epsilon) if num_patches > 1 else None
        )
        self.feature_mixer: TTMFeatureMixer = TTMFeatureMixer(size, expansion_factor, dropout, is_gated, epsilon)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Mix a batch of patch sequences.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, num_patches, size).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        if self.patch_mixer is not None:
            hidden = self.patch_mixer(hidden)
        return self.feature_mixer(hidden)

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
