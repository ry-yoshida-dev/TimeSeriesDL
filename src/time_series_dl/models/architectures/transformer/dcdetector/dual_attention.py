import math
from typing import cast

import torch
from torch import Tensor, nn


class DualAttention(nn.Module):
    """
    One dual attention layer: the same query and key projections attend over the patch tokens (patch-wise view) and
    over the in-patch position tokens (in-patch view), and both attention maps are expanded to step-by-step
    associations of the whole window.

    Parameters
    ----------
    model_size : int
        Size of each token.
    num_heads : int
        Number of heads; must divide model_size.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If num_heads does not divide model_size.
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float) -> None:
        if model_size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide model_size {model_size}")
        super().__init__()
        self.num_heads: int = num_heads
        self.query: nn.Linear = nn.Linear(model_size, model_size)
        self.key: nn.Linear = nn.Linear(model_size, model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, patch_tokens: Tensor, in_patch_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the step associations of both views.

        Parameters
        ----------
        patch_tokens : Tensor
            One token per patch with shape (batch, num_patches, model_size).
        in_patch_tokens : Tensor
            One token per position within a patch with shape (batch, patch_size, model_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The patch-wise and the in-patch associations, each with shape (batch, num_heads, length, length) where
            length is num_patches * patch_size; step t belongs to patch t // patch_size at position t % patch_size.
        """
        num_patches: int = patch_tokens.shape[1]
        patch_size: int = in_patch_tokens.shape[1]
        patch_wise: Tensor = self._attend(patch_tokens)
        in_patch: Tensor = self._attend(in_patch_tokens)
        expanded_patch_wise: Tensor = patch_wise.repeat_interleave(patch_size, dim=-2).repeat_interleave(
            patch_size, dim=-1
        )
        return expanded_patch_wise, in_patch.repeat(1, 1, num_patches, num_patches)

    def _attend(self, tokens: Tensor) -> Tensor:
        batch: int = tokens.shape[0]
        length: int = tokens.shape[1]
        queries: Tensor = self.query(tokens).view(batch, length, self.num_heads, -1).transpose(1, 2)
        keys: Tensor = self.key(tokens).view(batch, length, self.num_heads, -1).transpose(1, 2)
        scores: Tensor = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
        weights: Tensor = self.dropout(torch.softmax(scores, dim=-1))
        return weights

    def __call__(self, patch_tokens: Tensor, in_patch_tokens: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(patch_tokens, in_patch_tokens))
