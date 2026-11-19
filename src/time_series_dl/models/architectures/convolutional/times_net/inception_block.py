from typing import cast

import torch
from torch import Tensor, nn


class InceptionBlock(nn.Module):
    """
    Average of 2D convolutions with square kernels of odd sizes 1, 3, 5, ... that keep the spatial size.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    num_kernels : int
        Number of parallel convolutions.
    """

    def __init__(self, input_channels: int, output_channels: int, num_kernels: int) -> None:
        super().__init__()
        self.kernels: nn.ModuleList = nn.ModuleList(
            nn.Conv2d(input_channels, output_channels, kernel_size=2 * index + 1, padding=index)
            for index in range(num_kernels)
        )
        for kernel in self.kernels:
            if isinstance(kernel, nn.Conv2d):
                _ = nn.init.kaiming_normal_(kernel.weight, mode="fan_out", nonlinearity="relu")
                if kernel.bias is not None:
                    _ = nn.init.zeros_(kernel.bias)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the convolutions and average them.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, height, width).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, height, width).
        """
        outputs: list[Tensor] = [kernel(values) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(dim=-1)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
