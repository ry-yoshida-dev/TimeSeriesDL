from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDISideInformation(nn.Module):
    """
    Side information of CSDI for every entry of a (features, time) array: the sinusoidal embedding of its time
    position, the learned embedding of its feature, and whether it is conditioned on (observed).

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration.
    num_features : int
        Number of features of the arrays.

    Raises
    ------
    ValueError
        If num_features is not positive.
    """

    def __init__(self, parameters: CSDIParameters, num_features: int) -> None:
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        super().__init__()
        self.feature_embedding: nn.Embedding = nn.Embedding(num_features, parameters.feature_embedding_size)
        self.frequencies: Tensor
        self.register_buffer(
            "frequencies",
            torch.exp(
                torch.arange(0, parameters.time_embedding_size, 2, dtype=torch.float32)
                * (-math.log(10000.0) / parameters.time_embedding_size)
            ),
            persistent=False,
        )
        self.size: int = parameters.time_embedding_size + parameters.feature_embedding_size + 1

    def forward(self, conditional_mask: Tensor) -> Tensor:
        """
        Compute the side information.

        Parameters
        ----------
        conditional_mask : Tensor
            One for conditioned entries and zero otherwise, with shape (batch, features, length).

        Returns
        -------
        Tensor
            The side information with shape (batch, size, features, length).
        """
        batch: int = conditional_mask.shape[0]
        features: int = conditional_mask.shape[1]
        length: int = conditional_mask.shape[2]
        positions: Tensor = torch.arange(length, device=conditional_mask.device, dtype=self.frequencies.dtype)
        angles: Tensor = positions.unsqueeze(-1) * self.frequencies
        # Sines and cosines interleaved, as in the Transformer's position encoding.
        time: Tensor = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)
        time_part: Tensor = time.T[:, None, :].expand(-1, features, -1)
        feature_part: Tensor = self.feature_embedding.weight.T[:, :, None].expand(-1, -1, length)
        shared: Tensor = torch.cat([time_part, feature_part], dim=0).to(conditional_mask.dtype)
        return torch.cat([shared.expand(batch, -1, -1, -1), conditional_mask.unsqueeze(1)], dim=1)

    def __call__(self, conditional_mask: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(conditional_mask))
