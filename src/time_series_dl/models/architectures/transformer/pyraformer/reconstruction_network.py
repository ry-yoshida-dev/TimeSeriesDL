from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import PyraformerEncoder

if TYPE_CHECKING:
    from .parameters import PyraformerParameters


class PyraformerReconstructionNetwork(ReconstructionNetwork):
    """
    Pyraformer encoder whose features of each step, concatenated over all scales, are mapped linearly to its values.

    Parameters
    ----------
    parameters : PyraformerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: PyraformerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.length: int = shape.length
        self.encoder: PyraformerEncoder = PyraformerEncoder(parameters, shape.length, shape.channels)
        self.projection: nn.Linear = nn.Linear(self.encoder.feature_size, shape.channels)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)[:, -self.length :]
        reconstructed: Tensor = self.projection(features)
        return reconstructed
