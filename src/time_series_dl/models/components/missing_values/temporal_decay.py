from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class TemporalDecay(nn.Module):
    """
    Decay rates ``exp(-max(0, W gap + b))`` in (0, 1] from the time gaps since the last observations (Che et al.,
    2018).

    Parameters
    ----------
    input_size : int
        Number of channels of the gaps.
    output_size : int
        Number of decay rates.
    is_diagonal : bool
        Whether each rate depends only on the gap of its own channel; input_size must then equal output_size.

    Raises
    ------
    ValueError
        If a size is not positive, or the sizes differ for a diagonal decay.
    """

    def __init__(self, input_size: int, output_size: int, is_diagonal: bool) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"input_size and output_size must be positive, got {input_size} and {output_size}")
        if is_diagonal and input_size != output_size:
            raise ValueError(f"a diagonal decay needs equal sizes, got {input_size} and {output_size}")
        super().__init__()
        bound: float = output_size**-0.5
        self.weight: nn.Parameter = nn.Parameter(torch.empty(output_size, input_size).uniform_(-bound, bound))
        self.bias: nn.Parameter = nn.Parameter(torch.empty(output_size).uniform_(-bound, bound))
        self.weight_mask: Tensor
        self.register_buffer(
            "weight_mask",
            torch.eye(input_size) if is_diagonal else torch.ones(output_size, input_size),
            persistent=False,
        )

    def forward(self, gaps: Tensor) -> Tensor:
        """
        Compute the decay rates.

        Parameters
        ----------
        gaps : Tensor
            Time gaps with shape (..., input_size).

        Returns
        -------
        Tensor
            Decay rates with shape (..., output_size).
        """
        return torch.exp(-functional.relu(functional.linear(gaps, self.weight * self.weight_mask, self.bias)))

    def __call__(self, gaps: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(gaps))
