from collections.abc import Sequence
from typing import Final, cast

import torch
from torch import Tensor, nn

MAX_EMBEDDING_SIZE: Final[int] = 50


class StaticCovariateEmbedding(nn.Module):
    """
    Static features of each series as one vector: the numeric features as given, followed by an embedding of each
    categorical feature.

    Each categorical feature with ``cardinality`` categories is embedded into ``min(50, (cardinality + 1) // 2)``
    dimensions, the rule of the GluonTS implementation of DeepAR.

    Parameters
    ----------
    static_channels : int
        Number of numeric static features.
    cardinalities : Sequence[int]
        Number of categories of each categorical feature.

    Raises
    ------
    ValueError
        If static_channels is negative or a cardinality is not positive.
    """

    def __init__(self, static_channels: int, cardinalities: Sequence[int]) -> None:
        if static_channels < 0:
            raise ValueError(f"static_channels must not be negative, got {static_channels}")
        if any(cardinality <= 0 for cardinality in cardinalities):
            raise ValueError(f"cardinalities must be positive, got {tuple(cardinalities)}")
        super().__init__()
        sizes: list[int] = [min(MAX_EMBEDDING_SIZE, (cardinality + 1) // 2) for cardinality in cardinalities]
        self.embeddings: nn.ModuleList = nn.ModuleList(
            nn.Embedding(cardinality, size) for cardinality, size in zip(cardinalities, sizes, strict=True)
        )
        self.output_size: int = static_channels + sum(sizes)

    def forward(self, static: Tensor, categories: Tensor) -> Tensor:
        """
        Embed the static features of a batch.

        Parameters
        ----------
        static : Tensor
            Numeric features with shape (batch, static_channels).
        categories : Tensor
            Integer codes with shape (batch, len(cardinalities)).

        Returns
        -------
        Tensor
            The features with shape (batch, output_size); zero columns when there is no static feature.
        """
        embedded: list[Tensor] = [
            cast(Tensor, embedding(categories[:, position])) for position, embedding in enumerate(self.embeddings)
        ]
        return torch.cat([static, *embedded], dim=-1)

    def __call__(self, static: Tensor, categories: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(static, categories))
