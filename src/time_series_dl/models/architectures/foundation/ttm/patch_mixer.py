from typing import cast

from torch import Tensor, nn

from .gated_attention import TTMGatedAttention
from .perceptron import TTMPerceptron


class TTMPatchMixer(nn.Module):
    """
    Patch mixing of TinyTimeMixer: after a layer normalization of the features, a perceptron and an optional gate
    mix the patches of each feature; the result is added to the input.

    Parameters
    ----------
    num_patches : int
        Number of patches.
    size : int
        Number of features per patch.
    expansion_factor : int
        Ratio of the perceptron's hidden size to num_patches.
    dropout : float
        Dropout probability.
    is_gated : bool
        Whether the gated attention follows the perceptron.
    epsilon : float
        Epsilon of the layer normalization.
    """

    def __init__(
        self, num_patches: int, size: int, expansion_factor: int, dropout: float, is_gated: bool, epsilon: float
    ) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(size, eps=epsilon)
        self.perceptron: TTMPerceptron = TTMPerceptron(num_patches, expansion_factor, dropout)
        self.gate: TTMGatedAttention | None = TTMGatedAttention(num_patches) if is_gated else None

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Mix the patches.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, num_patches, size).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        mixed: Tensor = self.perceptron(self.norm(hidden).transpose(-1, -2))
        if self.gate is not None:
            mixed = self.gate(mixed)
        return hidden + mixed.transpose(-1, -2)

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
