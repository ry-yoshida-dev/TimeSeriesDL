from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers.convolution_block import ConvolutionBlock
from ....interfaces import SequenceEncoder
from .inception_module import InceptionTimeModule

if TYPE_CHECKING:
    from .parameters import InceptionTimeParameters


class InceptionTimeSequenceEncoder(SequenceEncoder):
    """
    Stack of length-preserving Inception modules with a batch-normalized 1x1 shortcut around each group of modules.

    Parameters
    ----------
    parameters : InceptionTimeParameters
        The module configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: InceptionTimeParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.module_size
        self.residual_interval: int = parameters.residual_interval
        self.inception_modules: nn.ModuleList = nn.ModuleList(
            InceptionTimeModule(
                input_channels if position == 0 else parameters.module_size,
                parameters.num_filters,
                parameters.bottleneck_size,
                parameters.kernel_sizes,
            )
            for position in range(parameters.depth)
        )
        num_shortcuts: int = parameters.depth // parameters.residual_interval if parameters.is_residual else 0
        self.shortcuts: nn.ModuleList = nn.ModuleList(
            ConvolutionBlock(
                input_channels if group == 0 else parameters.module_size,
                parameters.module_size,
                1,
                is_activated=False,
            )
            for group in range(num_shortcuts)
        )

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = values.transpose(1, 2)
        residual: Tensor = features
        for position, module in enumerate(self.inception_modules):
            features = module(features)
            group: int = position // self.residual_interval
            if (position + 1) % self.residual_interval == 0 and group < len(self.shortcuts):
                shortcut: Tensor = self.shortcuts[group](residual)
                features = functional.relu(features + shortcut)
                residual = features
        return features.transpose(1, 2)
