from typing import cast

import torch
from torch import Tensor, nn


class LearnedTimeEmbedding(nn.Module):
    """
    Learned embedding of continuous times (Shukla and Marlin, 2021): one linear component and sinusoids of learned
    frequencies and phases, ``[w_0 t + b_0, sin(w_1 t + b_1), ..., sin(w_{d-1} t + b_{d-1})]``.

    Parameters
    ----------
    size : int
        Size of the embedding.

    Raises
    ------
    ValueError
        If the size is less than 2.
    """

    def __init__(self, size: int) -> None:
        if size < 2:
            raise ValueError(f"size must be at least 2, got {size}")
        super().__init__()
        self.linear: nn.Linear = nn.Linear(1, 1)
        self.periodic: nn.Linear = nn.Linear(1, size - 1)

    def forward(self, times: Tensor) -> Tensor:
        """
        Embed times.

        Parameters
        ----------
        times : Tensor
            Times of any shape.

        Returns
        -------
        Tensor
            The embeddings with shape (*times.shape, size).
        """
        expanded: Tensor = times.unsqueeze(-1)
        return torch.cat([self.linear(expanded), torch.sin(self.periodic(expanded))], dim=-1)

    def __call__(self, times: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(times))
