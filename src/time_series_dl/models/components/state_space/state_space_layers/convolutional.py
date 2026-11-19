from typing import override

import torch
from torch import Tensor, nn

from ..fft_convolution import FFTConvolution
from ..gated_linear_output import GatedLinearOutput
from ..ssm_kernel import SSMKernel
from ..state_space_layer import StateSpaceLayer


class ConvolutionalStateSpaceLayer(StateSpaceLayer):
    """
    Linear time-invariant state space model of each channel, applied as a long convolution with its kernel, plus a
    skip term, followed by the gated output stage of S4 and S4D.

    Parameters
    ----------
    kernel : SSMKernel
        Kernel of the state space model of each channel.
    size : int
        Number of channels.
    dropout : float
        Dropout probability of the output stage.
    """

    def __init__(self, kernel: SSMKernel, size: int, dropout: float) -> None:
        super().__init__()
        self.kernel: SSMKernel = kernel
        self.skip: nn.Parameter = nn.Parameter(torch.randn(size))
        self.output: GatedLinearOutput = GatedLinearOutput(size, dropout)

    @override
    def forward(self, values: Tensor) -> Tensor:
        channels_first: Tensor = values.transpose(1, 2)
        convolved: Tensor = FFTConvolution.apply(channels_first, self.kernel(values.shape[1]))
        mixed: Tensor = (convolved + channels_first * self.skip[:, None]).transpose(1, 2)
        return self.output(mixed)
