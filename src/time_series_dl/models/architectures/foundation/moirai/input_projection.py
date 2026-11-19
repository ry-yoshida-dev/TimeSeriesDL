from typing import cast

import torch
from torch import Tensor, nn


class MoiraiInputProjection(nn.Module):
    """
    Multi-patch-size input projection of Moirai: one linear map per patch size from the zero-padded patch (of the
    largest patch size) to the token embedding, of which only the first ``patch_size`` inputs are read.

    Parameters
    ----------
    patch_sizes : tuple[int, ...]
        The supported patch sizes.
    output_size : int
        Size of the token embeddings.
    """

    def __init__(self, patch_sizes: tuple[int, ...], output_size: int) -> None:
        super().__init__()
        self.patch_sizes: tuple[int, ...] = patch_sizes
        maximum: int = max(patch_sizes)
        self.weight: nn.Parameter = nn.Parameter(torch.empty(len(patch_sizes), output_size, maximum))
        self.bias: nn.Parameter = nn.Parameter(torch.empty(len(patch_sizes), output_size))
        self.is_read: Tensor
        self.register_buffer(
            "is_read", torch.arange(maximum)[None, None, :] < torch.tensor(patch_sizes)[:, None, None], persistent=False
        )
        for index, size in enumerate(patch_sizes):
            nn.init.kaiming_uniform_(self.weight.data[index, :, :size], a=5**0.5)
            nn.init.zeros_(self.weight.data[index, :, size:])
            nn.init.uniform_(self.bias.data[index], -(size**-0.5), size**-0.5)

    def forward(self, patches: Tensor, patch_size: int) -> Tensor:
        """
        Embed patches of one patch size.

        Parameters
        ----------
        patches : Tensor
            Patches with shape (..., max(patch_sizes)), zero beyond ``patch_size``.
        patch_size : int
            One of ``patch_sizes``.

        Returns
        -------
        Tensor
            Embeddings with shape (..., output_size).
        """
        index: int = self.patch_sizes.index(patch_size)
        weight: Tensor = self.weight[index] * self.is_read[index]
        return patches @ weight.T + self.bias[index]

    def __call__(self, patches: Tensor, patch_size: int) -> Tensor:
        return cast(Tensor, super().__call__(patches, patch_size))
