from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm


class TemporalResidualBlock(nn.Module):
    """
    Two weight-normalized dilated causal convolutions with a residual connection.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int
        Length of the convolution kernels.
    dilation : int
        Spacing of the kernel taps.
    dropout : float
        Dropout probability after each convolution.
    """

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.padding: int = (kernel_size - 1) * dilation
        self.first: nn.Module = weight_norm(nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation))
        self.second: nn.Module = weight_norm(
            nn.Conv1d(output_channels, output_channels, kernel_size, dilation=dilation)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.shortcut: nn.Module = (
            nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()
        )

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
            Features with shape (batch, output_channels, length); step t depends only on steps up to t.
        """
        hidden: Tensor = self.dropout(functional.relu(self.first(functional.pad(values, (self.padding, 0)))))
        hidden = self.dropout(functional.relu(self.second(functional.pad(hidden, (self.padding, 0)))))
        shortcut: Tensor = self.shortcut(values)
        return functional.relu(hidden + shortcut)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
