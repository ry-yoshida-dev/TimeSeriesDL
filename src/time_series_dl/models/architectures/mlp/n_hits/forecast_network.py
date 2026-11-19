from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from ..n_beats import BasisExpansionBlock, DoublyResidualStacking

if TYPE_CHECKING:
    from .parameters import NHITSParameters


class NHITSForecastNetwork(ForecastNetwork):
    """
    Doubly residual stacks of blocks that read a pooled window and interpolate a coarse forecast, applied to each
    target channel's own history.

    The backcast of every block covers each input step; pooling kernels longer than the window are shortened to it.

    Parameters
    ----------
    parameters : NHITSParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: NHITSParameters, shape: ForecastShape) -> None:
        super().__init__()
        backcast_basis: Tensor = torch.eye(shape.input_length)
        blocks: list[BasisExpansionBlock] = []
        for stack in parameters.stacks:
            kernel_size: int = min(stack.pooling_kernel_size, shape.input_length)
            num_coefficients: int = math.ceil(shape.horizon / stack.downsampling_factor)
            forecast_basis: Tensor = parameters.interpolation.build_basis(num_coefficients, shape.horizon)
            blocks.extend(
                BasisExpansionBlock(
                    parameters.pooling.build(kernel_size),
                    math.ceil(shape.input_length / kernel_size),
                    stack.num_layers,
                    stack.hidden_size,
                    backcast_basis,
                    forecast_basis,
                    shape.num_parameters,
                )
                for _ in range(stack.num_blocks)
            )
        self.stacking: DoublyResidualStacking = DoublyResidualStacking(blocks)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        outputs: Tensor = self.stacking(past.index_select(2, self.target_channels))
        return outputs
