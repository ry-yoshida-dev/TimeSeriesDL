from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....components.random_convolution import KernelPooling, KernelTransform
from ..mini_rocket import MiniRocketKernelTransform

if TYPE_CHECKING:
    from .parameters import MultiRocketParameters

_POOLINGS: tuple[KernelPooling, ...] = (
    KernelPooling.PROPORTION_OF_POSITIVE_VALUES,
    KernelPooling.MEAN_OF_POSITIVE_VALUES,
    KernelPooling.MEAN_OF_INDICES_OF_POSITIVE_VALUES,
    KernelPooling.LONGEST_STRETCH_OF_POSITIVE_VALUES,
)


class MultiRocketKernelTransform(KernelTransform):
    """
    MiniRocket kernels applied to the series and to its first difference, each output pooled by PPV, MPV, MIPV and
    LSPV.

    Each representation has its own dilations, channel subsets and biases, fitted to the first batch as in
    ``MiniRocketKernelTransform``.

    Parameters
    ----------
    parameters : MultiRocketParameters
        The transform configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If input_length is below 2, so there is no first difference, or input_channels is not positive.
    """

    def __init__(self, parameters: MultiRocketParameters, input_length: int, input_channels: int) -> None:
        if input_length < 2:
            raise ValueError(f"input_length must be at least 2 for the first difference, got {input_length}")
        super().__init__()
        num_biases: int = parameters.num_features // (2 * len(_POOLINGS))
        self.original: MiniRocketKernelTransform = MiniRocketKernelTransform(
            input_length, input_channels, num_biases, parameters.max_dilations_per_kernel, _POOLINGS
        )
        self.difference: MiniRocketKernelTransform = MiniRocketKernelTransform(
            input_length - 1, input_channels, num_biases, parameters.max_dilations_per_kernel, _POOLINGS
        )

    @property
    @override
    def num_features(self) -> int:
        return self.original.num_features + self.difference.num_features

    @override
    def forward(self, values: Tensor) -> Tensor:
        return torch.cat([self.original(values), self.difference(values.diff(dim=1))], dim=1)
