from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class DiffusionStepEmbedding(nn.Module):
    """
    Sinusoidal features of the diffusion step followed by two linear layers with SiLU activations, as in DiffWave,
    TimeGrad and CSDI.

    Half of the features are sines and half cosines of the step at frequencies spaced geometrically from 1 to 10^4.

    Parameters
    ----------
    feature_size : int
        Number of sinusoidal features; even and at least 4.
    projection_size : int
        Size of the output vector.

    Raises
    ------
    ValueError
        If feature_size is odd or below 4, or projection_size is not positive.
    """

    def __init__(self, feature_size: int, projection_size: int) -> None:
        if feature_size < 4 or feature_size % 2 != 0:
            raise ValueError(f"feature_size must be even and at least 4, got {feature_size}")
        if projection_size <= 0:
            raise ValueError(f"projection_size must be positive, got {projection_size}")
        super().__init__()
        half: int = feature_size // 2
        self.frequencies: Tensor
        self.register_buffer(
            "frequencies",
            torch.pow(10.0, torch.arange(half, dtype=torch.float32) * (4.0 / (half - 1))),
            persistent=False,
        )
        self.first: nn.Linear = nn.Linear(feature_size, projection_size)
        self.second: nn.Linear = nn.Linear(projection_size, projection_size)

    def forward(self, steps: Tensor) -> Tensor:
        """
        Embed diffusion steps.

        Parameters
        ----------
        steps : Tensor
            Zero-based diffusion steps with shape (batch,).

        Returns
        -------
        Tensor
            The embeddings with shape (batch, projection_size).
        """
        angles: Tensor = steps.to(self.frequencies.dtype).unsqueeze(-1) * self.frequencies
        features: Tensor = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return functional.silu(self.second(functional.silu(self.first(features))))

    def __call__(self, steps: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(steps))
