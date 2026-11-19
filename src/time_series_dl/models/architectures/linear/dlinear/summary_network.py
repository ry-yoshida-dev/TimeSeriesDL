from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .decomposition_linear import DecompositionLinear

if TYPE_CHECKING:
    from .parameters import DLinearParameters


class DLinearSummaryNetwork(SummaryNetwork):
    """
    Decomposition linear maps along time followed by a linear layer over the flattened result.

    Parameters
    ----------
    parameters : DLinearParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: DLinearParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.linear: DecompositionLinear = DecompositionLinear(
            parameters.kernel_size,
            shape.input_length,
            shape.input_length,
            shape.input_channels,
            parameters.is_individual,
        )
        self.projection: nn.Linear = nn.Linear(shape.input_length * shape.input_channels, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        outputs: Tensor = self.projection(self.linear(values).flatten(start_dim=1))
        return outputs
