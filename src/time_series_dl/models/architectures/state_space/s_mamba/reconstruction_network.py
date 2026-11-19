from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import SMambaEncoder

if TYPE_CHECKING:
    from .parameters import SMambaParameters


class SMambaReconstructionNetwork(ReconstructionNetwork):
    """
    Variate tokens, each mapped linearly back to the whole series of its channel.

    Parameters
    ----------
    parameters : SMambaParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: SMambaParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: SMambaEncoder = SMambaEncoder(parameters, shape.length)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.length)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(values)
        reconstructed: Tensor = self.projection(tokens)
        return reconstructed.transpose(1, 2)
