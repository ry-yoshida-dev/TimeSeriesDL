from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .embedding import TimesNetEmbedding
from .times_block import TimesBlock

if TYPE_CHECKING:
    from .parameters import TimesNetParameters


class TimesNetSummaryNetwork(SummaryNetwork):
    """
    Embedding and TimesBlocks followed by GELU, dropout and a linear map of all flattened steps to the output vector.

    Parameters
    ----------
    parameters : TimesNetParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: TimesNetParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.embedding: TimesNetEmbedding = TimesNetEmbedding(
            shape.input_channels, parameters.model_size, shape.input_length, parameters.dropout
        )
        self.blocks: nn.Sequential = nn.Sequential(*(TimesBlock(parameters) for _ in range(parameters.num_layers)))
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size * shape.input_length, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.blocks(self.embedding(values))
        activated: Tensor = self.dropout(self.activation(features))
        outputs: Tensor = self.projection(activated.flatten(start_dim=1))
        return outputs
