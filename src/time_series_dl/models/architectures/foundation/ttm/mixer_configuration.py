from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TTMMixerConfiguration:
    """
    Sizes of a stack of TinyTimeMixer mixer layers (the backbone or the decoder).

    Attributes
    ----------
    num_patches : int
        Number of patches.
    size : int
        Number of features per patch.
    num_layers : int
        Number of mixer layers, per adaptive patching level if there are levels.
    adaptive_patching_levels : int
        Number of adaptive patching levels; level ``i`` splits each patch into ``2 ** i`` patches with
        ``size / 2 ** i`` features. Zero for a plain stack.
    expansion_factor : int
        Ratio of the perceptrons' hidden sizes to their input sizes.
    dropout : float
        Dropout probability.
    is_gated : bool
        Whether the gated attention follows each perceptron.
    epsilon : float
        Epsilon of the layer normalizations.
    """

    num_patches: int
    size: int
    num_layers: int
    adaptive_patching_levels: int
    expansion_factor: int
    dropout: float
    is_gated: bool
    epsilon: float

    def compute_patch_factor(self, level: int) -> int:
        """
        Get by how much a level divides the patches, as the original code does.

        Parameters
        ----------
        level : int
            The adaptive patching level.

        Returns
        -------
        int
            ``2 ** level``, or 1 when that would leave at most 4 features per patch.
        """
        factor: int = 2**level
        return 1 if self.size // factor <= 4 else factor
