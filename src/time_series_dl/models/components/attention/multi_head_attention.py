from typing import cast

from torch import Tensor, nn

from .attention_mechanism import AttentionMechanism


class MultiHeadAttention(nn.Module):
    """
    Projections of queries, keys and values into heads around a pluggable attention mechanism.

    Keys and values have no projection when the mechanism reads only the queries.

    Parameters
    ----------
    mechanism : AttentionMechanism
        The attention within each head.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of heads; it divides model_size.

    Raises
    ------
    ValueError
        If a size is not positive or num_heads does not divide model_size.
    """

    def __init__(self, mechanism: AttentionMechanism, model_size: int, num_heads: int) -> None:
        if model_size <= 0 or num_heads <= 0 or model_size % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide model_size, got {num_heads} and {model_size}")
        super().__init__()
        self.mechanism: AttentionMechanism = mechanism
        self.num_heads: int = num_heads
        self.query_projection: nn.Linear = nn.Linear(model_size, model_size)
        self.key_projection: nn.Linear | None = None if mechanism.is_query_only else nn.Linear(model_size, model_size)
        self.value_projection: nn.Linear | None = None if mechanism.is_query_only else nn.Linear(model_size, model_size)
        self.output_projection: nn.Linear = nn.Linear(model_size, model_size)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
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

        Returns
        -------
        Tensor
            Output tokens with shape (batch, query_length, model_size).
        """
        projected_queries: Tensor = self._split(self.query_projection(queries))
        projected_keys: Tensor = projected_queries
        projected_values: Tensor = projected_queries
        if self.key_projection is not None and self.value_projection is not None:
            projected_keys = self._split(self.key_projection(keys))
            projected_values = self._split(self.value_projection(values))
        attended: Tensor = self.mechanism(projected_queries, projected_keys, projected_values)
        outputs: Tensor = self.output_projection(attended.transpose(1, 2).flatten(start_dim=2))
        return outputs

    def _split(self, tokens: Tensor) -> Tensor:
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.num_heads, -1).transpose(1, 2)

    def __call__(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(queries, keys, values))
