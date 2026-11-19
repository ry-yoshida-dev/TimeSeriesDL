import math
from typing import cast

import torch
from torch import Tensor, nn


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable multi-head attention of TFT: heads have their own queries and keys but share the values, and
    their outputs are averaged, so the mean attention weights explain the output.

    Parameters
    ----------
    model_size : int
        Size of the input and output vectors.
    num_heads : int
        Number of heads; each projects queries, keys and values to ``model_size // num_heads``.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If a size is not positive, model_size is not divisible by num_heads, or dropout is not in [0, 1).
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float) -> None:
        if model_size <= 0 or num_heads <= 0:
            raise ValueError(f"model_size and num_heads must be positive, got {model_size} and {num_heads}")
        if model_size % num_heads != 0:
            raise ValueError(f"model_size {model_size} must be divisible by num_heads {num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = model_size // num_heads
        self.query: nn.Linear = nn.Linear(model_size, model_size)
        self.key: nn.Linear = nn.Linear(model_size, model_size)
        self.value: nn.Linear = nn.Linear(model_size, self.head_size)
        self.output: nn.Linear = nn.Linear(self.head_size, model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, is_masked: Tensor) -> tuple[Tensor, Tensor]:
        """
        Attend from each query to the unmasked keys.

        Parameters
        ----------
        queries : Tensor
            Queries with shape (batch, query_length, model_size).
        keys : Tensor
            Keys, also the values, with shape (batch, key_length, model_size).
        is_masked : Tensor
            Boolean mask with shape (query_length, key_length) that is True where a query must not attend.

        Returns
        -------
        tuple[Tensor, Tensor]
            The outputs with shape (batch, query_length, model_size) and the attention weights averaged over the
            heads with shape (batch, query_length, key_length).
        """
        batch_size: int = queries.shape[0]
        projected_queries: Tensor = self.query(queries).reshape(batch_size, -1, self.num_heads, self.head_size)
        projected_keys: Tensor = self.key(keys).reshape(batch_size, -1, self.num_heads, self.head_size)
        scores: Tensor = torch.einsum("bqhd,bkhd->bhqk", projected_queries, projected_keys) / math.sqrt(self.head_size)
        weights: Tensor = torch.softmax(scores.masked_fill(is_masked, float("-inf")), dim=-1)
        values: Tensor = self.value(keys)
        mixed: Tensor = torch.einsum("bhqk,bkd->bqd", self.dropout(weights), values) / self.num_heads
        outputs: Tensor = self.output(mixed)
        return outputs, weights.mean(dim=1)

    def __call__(self, queries: Tensor, keys: Tensor, is_masked: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(queries, keys, is_masked))
