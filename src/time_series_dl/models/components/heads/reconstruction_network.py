from typing import override

from torch import Tensor, nn

from ...interfaces import ReconstructionNetwork, SequenceEncoder, SequenceShape


class EncodedReconstructionNetwork(ReconstructionNetwork):
    """
    Encoder followed by a linear map from each time step's features to its values.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoder; it must keep the length of its input.
    shape : SequenceShape
        The dimensions of the sequences.
    dropout : float
        Dropout probability before the output layer.

    Raises
    ------
    ValueError
        If the encoder changes the length of the sequence.
    """

    def __init__(self, encoder: SequenceEncoder, shape: SequenceShape, dropout: float) -> None:
        if encoder.output_length != shape.length:
            raise ValueError(
                f"reconstruction needs an encoder that keeps the length {shape.length}, got {encoder.output_length}"
            )
        super().__init__()
        self.encoder: SequenceEncoder = encoder
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.projection: nn.Linear = nn.Linear(encoder.output_size, shape.channels)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(self.dropout(features))
        return reconstructed
