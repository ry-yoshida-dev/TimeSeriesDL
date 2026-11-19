from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class DeStationaryAttention(nn.Module):
    """
    Multi-head de-stationary attention: ``softmax((Q K^T tau + 1 delta^T) / sqrt(d)) V`` per head, where Q and K come
    from stationarized series, tau rescales every score of a window and delta shifts the scores of each key.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of heads; it divides model_size.
    dropout : float
        Dropout probability of the attention weights.
    is_causal : bool
        Whether each query sees only the keys up to its own position.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float, is_causal: bool) -> None:
        if model_size <= 0 or num_heads <= 0 or model_size % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide model_size, got {num_heads} and {model_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.is_causal: bool = is_causal
        self.query_projection: nn.Linear = nn.Linear(model_size, model_size)
        self.key_projection: nn.Linear = nn.Linear(model_size, model_size)
        self.value_projection: nn.Linear = nn.Linear(model_size, model_size)
        self.output_projection: nn.Linear = nn.Linear(model_size, model_size)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, scale: Tensor, shift: Tensor | None) -> Tensor:
        """
        Attend from every query token to the key and value tokens.

        Parameters
        ----------
        queries : Tensor
            Query tokens with shape (batch, query_length, model_size).
        keys : Tensor
            Key tokens with shape (batch, key_length, model_size).
        values : Tensor
            Value tokens with shape (batch, key_length, model_size).
        scale : Tensor
            The positive factor tau with shape (batch, 1).
        shift : Tensor | None
            The factor delta of each key with shape (batch, key_length); None for no shift.

        Returns
        -------
        Tensor
            Output tokens with shape (batch, query_length, model_size).
        """
        projected_queries: Tensor = self._split(self.query_projection(queries))
        projected_keys: Tensor = self._split(self.key_projection(keys))
        projected_values: Tensor = self._split(self.value_projection(values))
        head_size: int = projected_queries.shape[-1]
        # Scaling the queries by tau scales every score; delta enters as an additive mask, pre-divided by sqrt(d)
        # because the attention divides only the dot products by it.
        scaled_queries: Tensor = projected_queries * scale.reshape(-1, 1, 1, 1)
        bias: Tensor | None = None
        if shift is not None:
            shift_bias: Tensor = (shift / head_size**0.5).reshape(shift.shape[0], 1, 1, shift.shape[1])
            bias = (
                shift_bias + self._causal_bias(queries.shape[1], keys.shape[1], queries)
                if self.is_causal
                else shift_bias
            )
        attended: Tensor = functional.scaled_dot_product_attention(
            scaled_queries,
            projected_keys,
            projected_values,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal and bias is None,
        )
        outputs: Tensor = self.output_projection(attended.transpose(1, 2).flatten(start_dim=2))
        return outputs

    def _split(self, tokens: Tensor) -> Tensor:
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.num_heads, -1).transpose(1, 2)

    @staticmethod
    def _causal_bias(query_length: int, key_length: int, reference: Tensor) -> Tensor:
        is_future: Tensor = torch.ones(query_length, key_length, dtype=torch.bool, device=reference.device).triu(1)
        return torch.zeros(query_length, key_length, dtype=reference.dtype, device=reference.device).masked_fill(
            is_future, float("-inf")
        )

    def __call__(self, queries: Tensor, keys: Tensor, values: Tensor, scale: Tensor, shift: Tensor | None) -> Tensor:
        return cast(Tensor, super().__call__(queries, keys, values, scale, shift))
