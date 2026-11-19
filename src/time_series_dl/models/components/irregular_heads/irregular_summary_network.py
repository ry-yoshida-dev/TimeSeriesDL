from typing import override

from torch import Tensor, nn

from ...interfaces.irregular_sequence_encoder import IrregularSequenceEncoder
from ...interfaces.irregular_summary_network import IrregularSummaryNetwork
from ...interfaces.irregular_tensors import IrregularTensors


class EncodedIrregularSummaryNetwork(IrregularSummaryNetwork):
    """
    Feature vector of an irregular sequence encoder mapped to the output vector by output layers.

    Parameters
    ----------
    encoder : IrregularSequenceEncoder
        The encoder.
    output_layers : nn.Module
        Maps features with shape (batch, encoder.output_size) to outputs with shape (batch, output_size).
    """

    def __init__(self, encoder: IrregularSequenceEncoder, output_layers: nn.Module) -> None:
        super().__init__()
        self.encoder: IrregularSequenceEncoder = encoder
        self.output_layers: nn.Module = output_layers

    @override
    def forward(self, sequences: IrregularTensors) -> Tensor:
        outputs: Tensor = self.output_layers(self.encoder(sequences))
        return outputs
