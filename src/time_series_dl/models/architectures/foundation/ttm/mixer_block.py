from typing import cast

from torch import Tensor, nn

from .mixer_configuration import TTMMixerConfiguration
from .mixer_layer import TTMMixerLayer


class TTMMixerBlock(nn.Module):
    """
    Mixer layers at one adaptive patching level: each patch is split into ``patch_factor`` consecutive chunks of its
    features, which the layers mix as patches of their own, and merged back afterwards.

    Parameters
    ----------
    configuration : TTMMixerConfiguration
        The sizes of the stack.
    patch_factor : int
        Number of chunks each patch is split into; must divide the size.
    """

    def __init__(self, configuration: TTMMixerConfiguration, patch_factor: int) -> None:
        super().__init__()
        self.patch_factor: int = patch_factor
        self.layers: nn.ModuleList = nn.ModuleList(
            TTMMixerLayer(
                configuration.num_patches * patch_factor,
                configuration.size // patch_factor,
                configuration.expansion_factor,
                configuration.dropout,
                configuration.is_gated,
                configuration.epsilon,
            )
            for _ in range(configuration.num_layers)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Mix a batch of patch sequences at this level.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, num_patches, size).

        Returns
        -------
        Tensor
            The new hidden states with the same shape.
        """
        batch, num_patches, size = hidden.shape
        hidden = hidden.reshape(batch, num_patches * self.patch_factor, size // self.patch_factor)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.reshape(batch, num_patches, size)

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
