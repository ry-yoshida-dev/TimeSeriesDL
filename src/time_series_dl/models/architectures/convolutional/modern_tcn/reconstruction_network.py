from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import ModernTCNEncoder

if TYPE_CHECKING:
    from .parameters import ModernTCNParameters


class ModernTCNReconstructionNetwork(ReconstructionNetwork):
    """
    ModernTCN backbone, with each variable's flattened features mapped linearly back to its sequence.

    Parameters
    ----------
    parameters : ModernTCNParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: ModernTCNParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: ModernTCNEncoder = ModernTCNEncoder(parameters, shape.length, shape.channels)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(self.encoder.output_size * self.encoder.output_length, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(self.dropout(features.flatten(start_dim=2)))
        return reconstructed.transpose(1, 2)
