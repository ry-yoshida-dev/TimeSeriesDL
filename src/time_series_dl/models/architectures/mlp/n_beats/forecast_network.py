from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .basis_expansion_block import BasisExpansionBlock
from .doubly_residual_stacking import DoublyResidualStacking

if TYPE_CHECKING:
    from .parameters import NBEATSParameters


class NBEATSForecastNetwork(ForecastNetwork):
    """
    Doubly residual stacks of basis expansion blocks, applied to each target channel's own history.

    With several distribution parameters, each block expands separate coefficients for each parameter with the
    basis of its stack.

    Parameters
    ----------
    parameters : NBEATSParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: NBEATSParameters, shape: ForecastShape) -> None:
        super().__init__()
        blocks: list[BasisExpansionBlock] = []
        for stack in parameters.stacks:
            backcast_basis: Tensor
            forecast_basis: Tensor
            backcast_basis, forecast_basis = stack.kind.build_bases(
                shape.input_length, shape.horizon, parameters.polynomial_degree, parameters.num_harmonics
            )
            shared: BasisExpansionBlock | None = None
            for _ in range(stack.num_blocks):
                block: BasisExpansionBlock = (
                    shared
                    if shared is not None
                    else BasisExpansionBlock(
                        nn.Identity(),
                        shape.input_length,
                        stack.num_layers,
                        stack.hidden_size,
                        backcast_basis,
                        forecast_basis,
                        shape.num_parameters,
                    )
                )
                if stack.is_weight_shared:
                    shared = block
                blocks.append(block)
        self.stacking: DoublyResidualStacking = DoublyResidualStacking(blocks)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        outputs: Tensor = self.stacking(past.index_select(2, self.target_channels))
        return outputs
