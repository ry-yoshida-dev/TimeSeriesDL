from typing import cast

import torch
from torch import Tensor, nn


class MixHopPropagation(nn.Module):
    """
    Mix-hop propagation layer of MTGNN.

    The adjacency gains self-loops and is normalized by its row sums into ``P``; the hops
    ``H(k) = retain_ratio * H(0) + (1 - retain_ratio) * P @ H(k - 1)`` keep part of the input at every depth, and a
    pointwise convolution selects among the concatenated hops ``H(0), ..., H(depth)``.

    Parameters
    ----------
    input_channels : int
        Number of input features per node and step.
    output_channels : int
        Number of output features per node and step.
    depth : int
        Number of propagation steps.
    retain_ratio : float
        Share of the input kept at every step, in [0, 1].

    Raises
    ------
    ValueError
        If a size is not positive, depth is negative, or retain_ratio is not in [0, 1].
    """

    def __init__(self, input_channels: int, output_channels: int, depth: int, retain_ratio: float) -> None:
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError(
                f"input_channels and output_channels must be positive, got {input_channels} and {output_channels}"
            )
        if depth < 0:
            raise ValueError(f"depth must not be negative, got {depth}")
        if not 0.0 <= retain_ratio <= 1.0:
            raise ValueError(f"retain_ratio must be in [0, 1], got {retain_ratio}")
        super().__init__()
        self.depth: int = depth
        self.retain_ratio: float = retain_ratio
        self.selection: nn.Conv2d = nn.Conv2d((depth + 1) * input_channels, output_channels, kernel_size=1)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Propagate features over the graph.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, input_channels, nodes, length).
        adjacency : Tensor
            Non-negative adjacency with shape (nodes, nodes); entry (i, j) weighs the message node i receives from
            node j.

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, nodes, length).
        """
        looped: Tensor = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
        transition: Tensor = looped / looped.sum(dim=1, keepdim=True)
        hidden: Tensor = features
        hops: list[Tensor] = [features]
        for _ in range(self.depth):
            propagated: Tensor = torch.einsum("vw,bcwt->bcvt", transition, hidden)
            hidden = self.retain_ratio * features + (1.0 - self.retain_ratio) * propagated
            hops.append(hidden)
        selected: Tensor = self.selection(torch.cat(hops, dim=1))
        return selected

    def __call__(self, features: Tensor, adjacency: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features, adjacency))
