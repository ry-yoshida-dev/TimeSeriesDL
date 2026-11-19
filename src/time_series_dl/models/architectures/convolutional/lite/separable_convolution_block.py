from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class LITESeparableConvolutionBlock(nn.Module):
    """
    Depthwise separable convolution (a dilated depthwise convolution, then a pointwise convolution), batch
    normalization and a ReLU, keeping the length with the padding of Keras' ``same`` mode.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int
        Length of the depthwise kernel.
    dilation : int
        Dilation of the depthwise kernel.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        span: int = (kernel_size - 1) * dilation
        self.padding: tuple[int, int] = (span // 2, span - span // 2)
        self.depthwise: nn.Conv1d = nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size,
            dilation=dilation,
            groups=input_channels,
            bias=False,
        )
        self.pointwise: nn.Conv1d = nn.Conv1d(input_channels, output_channels, 1, bias=False)
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(output_channels)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, length).
        """
        mixed: Tensor = self.depthwise(functional.pad(values, self.padding))
        return functional.relu(self.normalization(self.pointwise(mixed)))

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
