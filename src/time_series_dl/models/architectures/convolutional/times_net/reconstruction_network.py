from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .embedding import TimesNetEmbedding
from .times_block import TimesBlock

if TYPE_CHECKING:
    from .parameters import TimesNetParameters


class TimesNetReconstructionNetwork(ReconstructionNetwork):
    """
    Embedding, TimesBlocks and a projection of each step's features back to its values.

    Parameters
    ----------
    parameters : TimesNetParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: TimesNetParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.embedding: TimesNetEmbedding = TimesNetEmbedding(
            shape.channels, parameters.model_size, shape.length, parameters.dropout
        )
        self.blocks: nn.Sequential = nn.Sequential(*(TimesBlock(parameters) for _ in range(parameters.num_layers)))
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.channels)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.blocks(self.embedding(values))
        reconstructed: Tensor = self.projection(features)
        return reconstructed
