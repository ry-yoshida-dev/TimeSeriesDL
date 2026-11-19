import math
from typing import cast

import torch
from torch import Tensor, nn


class T5RelativePositionBias(nn.Module):
    """
    Learned attention bias of each head for each bucket of relative positions (key position minus query position):
    exact buckets for short distances and logarithmically wider ones up to ``max_distance``.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    num_buckets : int
        Number of buckets; split between both directions when bidirectional.
    max_distance : int
        Distance beyond which positions share the last bucket.
    is_bidirectional : bool
        Whether keys after the query get their own buckets (encoder) or share the bucket of distance zero (decoder).
    """

    def __init__(self, num_heads: int, num_buckets: int, max_distance: int, is_bidirectional: bool) -> None:
        super().__init__()
        self.num_buckets: int = num_buckets
        self.max_distance: int = max_distance
        self.is_bidirectional: bool = is_bidirectional
        self.embedding: nn.Embedding = nn.Embedding(num_buckets, num_heads)

    def compute_buckets(self, relative_position: Tensor) -> Tensor:
        """
        Assign relative positions to buckets.

        Parameters
        ----------
        relative_position : Tensor
            Integer key positions minus query positions.

        Returns
        -------
        Tensor
            Bucket indices of the same shape.
        """
        num_buckets: int = self.num_buckets
        buckets: Tensor = torch.zeros_like(relative_position)
        if self.is_bidirectional:
            num_buckets //= 2
            buckets = buckets + (relative_position > 0).to(torch.long) * num_buckets
            distance: Tensor = relative_position.abs()
        else:
            distance = -torch.minimum(relative_position, torch.zeros_like(relative_position))
        max_exact: int = num_buckets // 2
        is_small: Tensor = distance < max_exact
        # Distances of zero are only reached in the exact branch; clamping avoids log(0) in the other.
        logarithmic: Tensor = max_exact + (
            torch.log(distance.clamp(min=1).to(torch.float32) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        logarithmic = torch.minimum(logarithmic, torch.full_like(logarithmic, num_buckets - 1))
        return buckets + torch.where(is_small, distance, logarithmic)

    def forward(self, query_length: int, key_length: int) -> Tensor:
        """
        Compute the bias of every query and key position.

        Parameters
        ----------
        query_length : int
            Number of query positions.
        key_length : int
            Number of key positions.

        Returns
        -------
        Tensor
            The bias with shape (1, num_heads, query_length, key_length).
        """
        device: torch.device = self.embedding.weight.device
        query_position: Tensor = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        key_position: Tensor = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        bias: Tensor = self.embedding(self.compute_buckets(key_position - query_position))
        return bias.permute(2, 0, 1).unsqueeze(0)

    def __call__(self, query_length: int, key_length: int) -> Tensor:
        return cast(Tensor, super().__call__(query_length, key_length))
