from typing import cast

import torch
from torch import Tensor, nn


class MoiraiOutputProjection(nn.Module):
    """
    Multi-patch-size output projection of Moirai: one linear map per patch size from a token representation to
    ``size_per_value`` values per position of the largest patch size, laid out as in the published weights (the
    first ``size_per_value * patch_size`` outputs are learned, the others are bias only), and returned with shape
    (..., max(patch_sizes), size_per_value).

    Parameters
    ----------
    input_size : int
        Size of the token representations.
    patch_sizes : tuple[int, ...]
        The supported patch sizes.
    size_per_value : int
        Number of outputs per value of a patch.
    """

    def __init__(self, input_size: int, patch_sizes: tuple[int, ...], size_per_value: int) -> None:
        super().__init__()
        self.patch_sizes: tuple[int, ...] = patch_sizes
        self.size_per_value: int = size_per_value
        self.maximum_patch_size: int = max(patch_sizes)
        maximum: int = size_per_value * self.maximum_patch_size
        self.weight: nn.Parameter = nn.Parameter(torch.empty(len(patch_sizes), maximum, input_size))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(len(patch_sizes), maximum))
        sizes: Tensor = torch.tensor(patch_sizes) * size_per_value
        self.is_written: Tensor
        self.register_buffer(
            "is_written", torch.arange(maximum)[None, :, None] < sizes[:, None, None], persistent=False
        )
        for index, size in enumerate(patch_sizes):
            nn.init.kaiming_uniform_(self.weight.data[index], a=5**0.5)
            self.weight.data[index, size * size_per_value :] = 0.0
            nn.init.uniform_(self.bias.data[index, : size * size_per_value], -(input_size**-0.5), input_size**-0.5)

    def forward(self, hidden: Tensor, patch_size: int) -> Tensor:
        """
        Project token representations of one patch size.

        Parameters
        ----------
        hidden : Tensor
            Representations with shape (..., input_size).
        patch_size : int
            One of ``patch_sizes``.

        Returns
        -------
        Tensor
            Outputs with shape (..., max(patch_sizes), size_per_value).
        """
        index: int = self.patch_sizes.index(patch_size)
        weight: Tensor = self.weight[index] * self.is_written[index]
        projected: Tensor = hidden @ weight.T + self.bias[index]
        # The published layout is value-major: output (value d, position t) sits at d * max_patch + t.
        split: Tensor = projected.reshape(*projected.shape[:-1], self.size_per_value, self.maximum_patch_size)
        return split.transpose(-1, -2)

    def __call__(self, hidden: Tensor, patch_size: int) -> Tensor:
        return cast(Tensor, super().__call__(hidden, patch_size))
