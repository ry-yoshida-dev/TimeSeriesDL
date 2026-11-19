from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....interfaces import ReconstructionNetwork, SequenceShape
from .decomposition_linear import DecompositionLinear

if TYPE_CHECKING:
    from .parameters import DLinearParameters


class DLinearReconstructionNetwork(ReconstructionNetwork):
    """
    Decomposition into trend and seasonal parts, each mapped linearly back onto the window.

    Parameters
    ----------
    parameters : DLinearParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: DLinearParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.linear: DecompositionLinear = DecompositionLinear(
            parameters.kernel_size, shape.length, shape.length, shape.channels, parameters.is_individual
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        return self.linear(values)
