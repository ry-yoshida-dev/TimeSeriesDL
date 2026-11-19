from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class AttentionMechanism(nn.Module, ABC):
    """
    Mixes the values of a sequence for each query, within each head of a multi-head attention.
    """

    @property
    def is_query_only(self) -> bool:
        """
        Check whether the mechanism reads only the queries, so keys and values need no projection.

        Returns
        -------
        bool
            True if keys and values are ignored.
        """
        return False

    @abstractmethod
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """
        Attend from every query to the keys and values.

        Parameters
        ----------
        queries : Tensor
            Queries with shape (batch, heads, query_length, head_size).
        keys : Tensor
            Keys with shape (batch, heads, key_length, head_size).
        values : Tensor
            Values with shape (batch, heads, key_length, head_size).

        Returns
        -------
        Tensor
            Outputs with shape (batch, heads, query_length, head_size).
        """

    def __call__(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(queries, keys, values))
