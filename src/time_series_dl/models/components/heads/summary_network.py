from typing import override

from torch import Tensor, nn

from ...interfaces import SequenceEncoder, SummaryNetwork, SummaryShape
from ..layers import TemporalPooling


class EncodedSummaryNetwork(SummaryNetwork):
    """
    Encoder followed by temporal pooling and a linear output layer.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoder of the sequence.
    shape : SummaryShape
        The input and output dimensions.
    pooling : TemporalPooling
        Reduction of the encoded sequence to one vector.
    dropout : float
        Dropout probability before the output layer.
    """

    def __init__(self, encoder: SequenceEncoder, shape: SummaryShape, pooling: TemporalPooling, dropout: float) -> None:
        super().__init__()
        self.encoder: SequenceEncoder = encoder
        self.pooling: TemporalPooling = pooling
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.projection: nn.Linear = nn.Linear(
            pooling.output_size(encoder.output_length, encoder.output_size), shape.output_size
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        outputs: Tensor = self.projection(self.dropout(self.pooling.apply(features)))
        return outputs
