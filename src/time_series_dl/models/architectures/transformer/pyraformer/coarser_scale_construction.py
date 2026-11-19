from typing import cast

import torch
from torch import Tensor, nn


class PyraformerCoarserScaleConstruction(nn.Module):
    """
    Coarser-scale construction module with a bottleneck: the tokens are projected down, each coarser scale is a
    strided convolution of the scale below followed by batch normalization and ELU, the coarser scales are projected
    back up, appended to the input tokens, and all nodes are layer-normalized.

    Parameters
    ----------
    model_size : int
        Size of each token.
    bottleneck_size : int
        Size of the tokens inside the convolutions.
    window_sizes : tuple[int, ...]
        Kernel size and stride of the convolution of each coarser scale, from the finest.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, model_size: int, bottleneck_size: int, window_sizes: tuple[int, ...]) -> None:
        if model_size <= 0 or bottleneck_size <= 0 or any(size <= 0 for size in window_sizes):
            raise ValueError(
                "model_size, bottleneck_size and window_sizes must be positive, got "
                + f"{model_size}, {bottleneck_size} and {window_sizes}"
            )
        super().__init__()
        self.down: nn.Linear = nn.Linear(model_size, bottleneck_size)
        self.convolutions: nn.ModuleList = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(bottleneck_size, bottleneck_size, kernel_size=window_size, stride=window_size),
                nn.BatchNorm1d(bottleneck_size),
                nn.ELU(),
            )
            for window_size in window_sizes
        )
        self.up: nn.Linear = nn.Linear(bottleneck_size, model_size)
        self.norm: nn.LayerNorm = nn.LayerNorm(model_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Build the nodes of every scale.

        Parameters
        ----------
        tokens : Tensor
            Tokens of the input steps with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Nodes of all scales, from the finest, with shape (batch, num_nodes, model_size).
        """
        scale: Tensor = self.down(tokens).transpose(1, 2)
        coarser: list[Tensor] = []
        for convolution in self.convolutions:
            scale = convolution(scale)
            coarser.append(scale)
        nodes: Tensor = torch.cat([tokens, self.up(torch.cat(coarser, dim=2).transpose(1, 2))], dim=1)
        normalized: Tensor = self.norm(nodes)
        return normalized

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
