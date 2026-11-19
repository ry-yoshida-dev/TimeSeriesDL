import math
from typing import cast

import torch
from torch import Tensor, nn


class TimesFMAttention(nn.Module):
    """
    Multi-head self-attention of TimesFM, whose queries are scaled per dimension by a learned softplus factor instead
    of the fixed inverse square root of the head size.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of heads.
    head_size : int
        Size of each head.
    """

    def __init__(self, model_size: int, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.query: nn.Linear = nn.Linear(model_size, num_heads * head_size)
        self.key: nn.Linear = nn.Linear(model_size, num_heads * head_size)
        self.value: nn.Linear = nn.Linear(model_size, num_heads * head_size)
        self.output: nn.Linear = nn.Linear(num_heads * head_size, model_size)
        self.query_scale: nn.Parameter = nn.Parameter(torch.zeros(head_size))

    def forward(self, hidden: Tensor, bias: Tensor) -> Tensor:
        """
        Attend from every position to the positions the bias leaves visible.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).
        bias : Tensor
            Additive bias broadcastable to (batch, num_heads, length, length), with large negative values at hidden
            keys.

        Returns
        -------
        Tensor
            The attended values with shape (batch, length, model_size).
        """
        # 1.442695041 = 1 / ln(2), so that softplus(0) scales by 1 / sqrt(head_size) as in the original.
        scale: Tensor = nn.functional.softplus(self.query_scale) * (1.442695041 / math.sqrt(self.head_size))
        query: Tensor = self._split_heads(self.query(hidden)) * scale
        key: Tensor = self._split_heads(self.key(hidden))
        value: Tensor = self._split_heads(self.value(hidden))
        scores: Tensor = query @ key.transpose(-1, -2) + bias
        weights: Tensor = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
        attended: Tensor = (weights @ value).transpose(1, 2).flatten(start_dim=2)
        outputs: Tensor = self.output(attended)
        return outputs

    def __call__(self, hidden: Tensor, bias: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, bias))

    def _split_heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        split: Tensor = projected.reshape(batch, length, self.num_heads, self.head_size).transpose(1, 2)
        return split
