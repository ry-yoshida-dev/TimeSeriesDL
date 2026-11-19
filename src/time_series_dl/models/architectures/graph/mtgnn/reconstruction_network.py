from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import MTGNNEncoder

if TYPE_CHECKING:
    from .parameters import MTGNNParameters


class MTGNNReconstructionNetwork(ReconstructionNetwork):
    """
    MTGNN node features, each mapped linearly back to the whole series of its channel.

    Parameters
    ----------
    parameters : MTGNNParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: MTGNNParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: MTGNNEncoder = MTGNNEncoder(parameters, shape.channels, shape.length)
        self.projection: nn.Linear = nn.Linear(parameters.end_channels, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(features)
        return reconstructed.transpose(1, 2)
