from typing import cast

import torch
from torch import Tensor, nn


class T5Attention(nn.Module):
    """
    Multi-head attention of T5: projections without bias and unscaled dot products, to which the relative position
    bias and the mask are added.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    key_value_size : int
        Size of each head.
    num_heads : int
        Number of heads.
    dropout : float
        Dropout probability of the attention weights.
    """

    def __init__(self, model_size: int, key_value_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        inner_size: int = num_heads * key_value_size
        self.num_heads: int = num_heads
        self.key_value_size: int = key_value_size
        self.query: nn.Linear = nn.Linear(model_size, inner_size, bias=False)
        self.key: nn.Linear = nn.Linear(model_size, inner_size, bias=False)
        self.value: nn.Linear = nn.Linear(model_size, inner_size, bias=False)
        self.output: nn.Linear = nn.Linear(inner_size, model_size, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, key_value: Tensor, bias: Tensor) -> Tensor:
        """
        Attend from every query position to every key position.

        Parameters
        ----------
        hidden : Tensor
            Queries with shape (batch, query_length, model_size).
        key_value : Tensor
            Keys and values with shape (batch, key_length, model_size); ``hidden`` itself for self-attention.
        bias : Tensor
            Additive bias with shape broadcastable to (batch, num_heads, query_length, key_length), holding the
            relative position bias and large negative values at masked positions.

        Returns
        -------
        Tensor
            The attended values with shape (batch, query_length, model_size).
        """
        key, value = self.project_keys_values(key_value)
        return self.attend(hidden, key, value, bias)

    def project_keys_values(self, key_value: Tensor) -> tuple[Tensor, Tensor]:
        """
        Project states to the keys and values of every head, so they can be cached and reused by ``attend``.

        Parameters
        ----------
        key_value : Tensor
            States with shape (batch, key_length, model_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The keys and the values, each with shape (batch, num_heads, key_length, key_value_size).
        """
        return self._split_heads(self.key(key_value)), self._split_heads(self.value(key_value))

    def attend(self, hidden: Tensor, key: Tensor, value: Tensor, bias: Tensor) -> Tensor:
        """
        Attend from queries to keys and values projected beforehand; equal to ``forward`` with the same states.

        Parameters
        ----------
        hidden : Tensor
            Queries with shape (batch, query_length, model_size).
        key : Tensor
            Keys with shape (batch, num_heads, key_length, key_value_size).
        value : Tensor
            Values with shape (batch, num_heads, key_length, key_value_size).
        bias : Tensor
            Additive bias broadcastable to (batch, num_heads, query_length, key_length).

        Returns
        -------
        Tensor
            The attended values with shape (batch, query_length, model_size).
        """
        query: Tensor = self._split_heads(self.query(hidden))
        scores: Tensor = query @ key.transpose(-1, -2) + bias
        weights: Tensor = self.dropout(torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype))
        attended: Tensor = (weights @ value).transpose(1, 2).flatten(start_dim=2)
        outputs: Tensor = self.output(attended)
        return outputs

    def _split_heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        split: Tensor = projected.reshape(batch, length, self.num_heads, self.key_value_size).transpose(1, 2)
        return split

    def __call__(self, hidden: Tensor, key_value: Tensor, bias: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, key_value, bias))
