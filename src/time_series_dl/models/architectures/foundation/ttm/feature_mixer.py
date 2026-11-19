from typing import cast

from torch import Tensor, nn

from .gated_attention import TTMGatedAttention
from .perceptron import TTMPerceptron


class TTMFeatureMixer(nn.Module):
    """
    Feature mixing of TinyTimeMixer: a layer normalization, a perceptron and an optional gate over the features of
    each patch; the result is added to the input.

    Parameters
    ----------
    size : int
        Number of features per patch.
    expansion_factor : int
        Ratio of the perceptron's hidden size to size.
    dropout : float
        Dropout probability.
    is_gated : bool
        Whether the gated attention follows the perceptron.
    epsilon : float
        Epsilon of the layer normalization.
    """

    def __init__(self, size: int, expansion_factor: int, dropout: float, is_gated: bool, epsilon: float) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(size, eps=epsilon)
        self.perceptron: TTMPerceptron = TTMPerceptron(size, expansion_factor, dropout)
        self.gate: TTMGatedAttention | None = TTMGatedAttention(size) if is_gated else None

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Mix the features.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, num_patches, size).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        mixed: Tensor = self.perceptron(self.norm(hidden))
        if self.gate is not None:
            mixed = self.gate(mixed)
        return hidden + mixed

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
