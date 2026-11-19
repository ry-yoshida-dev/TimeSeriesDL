from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class SameLengthConvolution(nn.Module):
    """
    One-dimensional convolution padded with zeros so the output keeps the length of the input.

    Even kernels are padded by one more step at the end than at the start.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int
        Length of the kernel.
    is_biased : bool
        Whether the convolution adds a learned bias.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, is_biased: bool = True) -> None:
        if input_channels <= 0 or output_channels <= 0 or kernel_size <= 0:
            raise ValueError(
                "input_channels, output_channels and kernel_size must be positive, got "
                + f"{input_channels}, {output_channels} and {kernel_size}"
            )
        super().__init__()
        self.padding: tuple[int, int] = ((kernel_size - 1) // 2, kernel_size // 2)
        self.convolution: nn.Conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, bias=is_biased)

    def forward(self, values: Tensor) -> Tensor:
        """
        Convolve a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, length).
        """
        features: Tensor = self.convolution(functional.pad(values, self.padding))
        return features

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
