import math
from typing import cast

import torch
from torch import Tensor, nn


class MOMENTPatchEmbedding(nn.Module):
    """
    Patch embedding of MOMENT: a linear map of each visible patch, a learned mask embedding in place of hidden
    patches, and a fixed sinusoidal position embedding.

    Parameters
    ----------
    patch_length : int
        Number of values per patch.
    model_size : int
        Size of the embeddings.
    dropout : float
        Dropout probability of the embeddings.
    is_positional_embedding_used : bool
        Whether the sinusoidal position embedding is added.
    num_patches : int
        Number of patches per series, the length of the position table.
    """

    def __init__(
        self,
        patch_length: int,
        model_size: int,
        dropout: float,
        is_positional_embedding_used: bool,
        num_patches: int,
    ) -> None:
        super().__init__()
        self.value_embedding: nn.Linear = nn.Linear(patch_length, model_size, bias=False)
        _ = nn.init.orthogonal_(self.value_embedding.weight, gain=1.41)
        self.mask_embedding: nn.Parameter = nn.Parameter(torch.zeros(model_size))
        self.is_positional_embedding_used: bool = is_positional_embedding_used
        position: Tensor = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
        frequency: Tensor = (
            torch.arange(0, model_size, 2, dtype=torch.float32) * -(math.log(10000.0) / model_size)
        ).exp()
        table: Tensor = torch.zeros(num_patches, model_size)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.position_table: Tensor
        self.register_buffer("position_table", table, persistent=False)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, patches: Tensor, is_visible: Tensor) -> Tensor:
        """
        Embed a batch of patched series.

        Parameters
        ----------
        patches : Tensor
            Patches with shape (batch, num_patches, patch_length).
        is_visible : Tensor
            Boolean mask with shape (batch, num_patches); hidden patches get the mask embedding.

        Returns
        -------
        Tensor
            Embeddings with shape (batch, num_patches, model_size).
        """
        embedded: Tensor = torch.where(is_visible.unsqueeze(-1), self.value_embedding(patches), self.mask_embedding)
        if self.is_positional_embedding_used:
            embedded = embedded + self.position_table[: patches.shape[1]]
        outputs: Tensor = self.dropout(embedded)
        return outputs

    def __call__(self, patches: Tensor, is_visible: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(patches, is_visible))
