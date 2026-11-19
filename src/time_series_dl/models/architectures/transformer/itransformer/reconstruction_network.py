from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import ITransformerEncoder

if TYPE_CHECKING:
    from .parameters import ITransformerParameters


class ITransformerReconstructionNetwork(ReconstructionNetwork):
    """
    Variate tokens, each mapped linearly back to the whole series of its channel.

    Parameters
    ----------
    parameters : ITransformerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: ITransformerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: ITransformerEncoder = ITransformerEncoder(parameters, shape.length)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(tokens)
        return reconstructed.transpose(1, 2)
