from typing import cast

from torch import Tensor, nn

from .mixer_block import TTMMixerBlock
from .mixer_configuration import TTMMixerConfiguration


class TTMMixerStack(nn.Module):
    """
    Stack of TinyTimeMixer mixer layers: with adaptive patching, one block per level from the finest split to no
    split; otherwise one block of all layers without splitting.

    Parameters
    ----------
    configuration : TTMMixerConfiguration
        The sizes of the stack.
    """

    def __init__(self, configuration: TTMMixerConfiguration) -> None:
        super().__init__()
        levels: list[int] = list(reversed(range(configuration.adaptive_patching_levels))) or [0]
        self.blocks: nn.ModuleList = nn.ModuleList(
            TTMMixerBlock(configuration, configuration.compute_patch_factor(level)) for level in levels
        )

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
            The new hidden states with the same shape.
        """
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
