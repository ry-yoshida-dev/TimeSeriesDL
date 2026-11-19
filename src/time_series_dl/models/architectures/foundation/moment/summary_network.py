from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import MOMENTEncoder
from .encoding import MOMENTEncoding

if TYPE_CHECKING:
    from .parameters import MOMENTParameters


class MOMENTSummaryNetwork(SummaryNetwork):
    """
    MOMENT's classification head: the patch states of each channel are averaged, the averages of all channels are
    concatenated (the official ``concat`` reduction), and dropout and a newly initialized linear layer give the
    output vector.

    Sequences shorter than ``input_length`` are left-padded.

    Parameters
    ----------
    parameters : MOMENTParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.

    Raises
    ------
    ValueError
        If the sequences are longer than ``input_length``.
    """

    def __init__(self, parameters: MOMENTParameters, shape: SummaryShape) -> None:
        if shape.input_length > parameters.input_length:
            raise ValueError(f"MOMENT reads at most {parameters.input_length} steps, got {shape.input_length}")
        super().__init__()
        self.encoder: MOMENTEncoder = MOMENTEncoder(parameters)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.head: nn.Linear = nn.Linear(shape.input_channels * parameters.model_size, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch: int = values.shape[0]
        padded, is_input = self.encoder.pad(values.transpose(1, 2).flatten(end_dim=1))
        encoding: MOMENTEncoding = self.encoder(padded, is_input, is_input, is_input)
        pooled: Tensor = encoding.states.mean(dim=1).reshape(batch, -1)
        outputs: Tensor = self.head(self.dropout(pooled))
        return outputs
