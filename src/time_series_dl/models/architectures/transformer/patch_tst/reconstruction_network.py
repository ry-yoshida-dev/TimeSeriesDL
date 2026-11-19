from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import PatchTSTEncoder

if TYPE_CHECKING:
    from .parameters import PatchTSTParameters


class PatchTSTReconstructionNetwork(ReconstructionNetwork):
    """
    Patch encoder of each channel followed by a linear map of its flattened features back to the whole sequence.

    Parameters
    ----------
    parameters : PatchTSTParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: PatchTSTParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: PatchTSTEncoder = PatchTSTEncoder(parameters, shape.length)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(self.encoder.feature_size, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(self.dropout(features.flatten(start_dim=2)))
        return reconstructed.transpose(1, 2)
