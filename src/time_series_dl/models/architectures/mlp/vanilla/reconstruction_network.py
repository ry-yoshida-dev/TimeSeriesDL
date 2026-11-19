from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....components.layers import MultilayerPerceptron
from ....interfaces import ReconstructionNetwork, SequenceShape

if TYPE_CHECKING:
    from .parameters import MLPParameters


class MLPReconstructionNetwork(ReconstructionNetwork):
    """
    Multilayer perceptron from the flattened sequence to its flattened reconstruction.

    Parameters
    ----------
    parameters : MLPParameters
        The layer configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: MLPParameters, shape: SequenceShape) -> None:
        super().__init__()
        size: int = shape.length * shape.channels
        self.layers: MultilayerPerceptron = MultilayerPerceptron(
            size, parameters.hidden_sizes, size, parameters.activation, parameters.dropout
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        outputs: Tensor = self.layers(values.flatten(start_dim=1))
        return outputs.reshape(values.shape)
