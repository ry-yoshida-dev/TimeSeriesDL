from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import CrossformerEncoder

if TYPE_CHECKING:
    from .parameters import CrossformerParameters


class CrossformerReconstructionNetwork(ReconstructionNetwork):
    """
    Crossformer encoder whose coarsest-scale tokens of each channel are flattened and mapped linearly, shared across
    channels, back to the channel's whole series, as in the Time-Series-Library.

    Parameters
    ----------
    parameters : CrossformerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: CrossformerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: CrossformerEncoder = CrossformerEncoder(parameters, shape.length, shape.channels)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(self.encoder.scale_segments[-1] * parameters.model_size, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(values)[-1]
        reconstructed: Tensor = self.projection(self.dropout(tokens.flatten(start_dim=2)))
        return reconstructed.transpose(1, 2)
