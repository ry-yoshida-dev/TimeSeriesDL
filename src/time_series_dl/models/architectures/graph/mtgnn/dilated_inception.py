from typing import cast

import torch
from torch import Tensor, nn


class MTGNNDilatedInception(nn.Module):
    """
    Dilated inception layer of MTGNN: dilated convolutions along time with kernels of 2, 3, 6 and 7 steps, each
    producing a quarter of the output channels, cropped to the shortest output and concatenated.

    Parameters
    ----------
    input_channels : int
        Number of input features per node and step.
    output_channels : int
        Number of output features per node and step; must be divisible by the number of kernels.
    dilation : int
        Spacing of the kernel taps.

    Raises
    ------
    ValueError
        If a size or the dilation is not positive, or output_channels is not divisible by the number of kernels.
    """

    KERNEL_SIZES: tuple[int, ...] = (2, 3, 6, 7)

    def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
        if input_channels <= 0 or output_channels <= 0 or dilation <= 0:
            raise ValueError(
                "input_channels, output_channels and dilation must be positive, got "
                + f"{input_channels}, {output_channels} and {dilation}"
            )
        if output_channels % len(self.KERNEL_SIZES) != 0:
            raise ValueError(f"output_channels must be divisible by {len(self.KERNEL_SIZES)}, got {output_channels}")
        super().__init__()
        branch_channels: int = output_channels // len(self.KERNEL_SIZES)
        self.branches: nn.ModuleList = nn.ModuleList(
            nn.Conv2d(input_channels, branch_channels, (1, kernel_size), dilation=(1, dilation))
            for kernel_size in self.KERNEL_SIZES
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Convolve features along time.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, input_channels, nodes, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, nodes, length - 6 * dilation).
        """
        outputs: list[Tensor] = [branch(features) for branch in self.branches]
        length: int = outputs[-1].shape[-1]
        return torch.cat([output[..., -length:] for output in outputs], dim=1)

    def __call__(self, features: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features))
