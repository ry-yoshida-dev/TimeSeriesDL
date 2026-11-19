from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .decoder_layer import TimerDecoderLayer

if TYPE_CHECKING:
    from .parameters import TimerParameters


class TimerModel(nn.Module):
    """
    Timer (Liu et al., 2024): a decoder-only Transformer over non-overlapping patches of a normalized series, in
    which each position predicts the next ``output_patch_length`` values from the patches up to it.

    Parameters
    ----------
    parameters : TimerParameters
        The model configuration.
    """

    def __init__(self, parameters: TimerParameters) -> None:
        super().__init__()
        self.patch_embedding: nn.Linear = nn.Linear(parameters.patch_length, parameters.model_size, bias=False)
        self.layers: nn.ModuleList = nn.ModuleList(TimerDecoderLayer(parameters) for _ in range(parameters.num_layers))
        self.final_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size, eps=parameters.layer_norm_epsilon)
        self.head: nn.Linear = nn.Linear(parameters.model_size, parameters.output_patch_length, bias=False)

    def forward(self, patches: Tensor) -> Tensor:
        """
        Predict the values after every patch.

        Parameters
        ----------
        patches : Tensor
            Normalized patches with shape (batch, num_patches, patch_length), oldest first.

        Returns
        -------
        Tensor
            Normalized predictions with shape (batch, num_patches, output_patch_length); position ``i`` predicts the
            values after patch ``i``.
        """
        hidden: Tensor = self.patch_embedding(patches)
        for layer in self.layers:
            hidden = layer(hidden)
        outputs: Tensor = self.head(self.final_norm(hidden))
        return outputs

    def __call__(self, patches: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(patches))
