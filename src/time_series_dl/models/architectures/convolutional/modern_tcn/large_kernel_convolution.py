from typing import cast

from torch import Tensor, nn


class ModernTCNLargeKernelConvolution(nn.Module):
    """
    Depthwise convolution with a large kernel plus one with a small kernel, each batch-normalized, keeping the length.

    This is the training form of the structural reparameterization of ModernTCN; the official code can merge the
    small kernel into the large one for inference, which gives the same outputs.

    Parameters
    ----------
    channels : int
        Number of channels, each convolved separately.
    large_kernel_size : int
        Odd length of the large kernel.
    small_kernel_size : int
        Odd length of the small kernel.
    """

    def __init__(self, channels: int, large_kernel_size: int, small_kernel_size: int) -> None:
        super().__init__()
        self.large: nn.Conv1d = nn.Conv1d(
            channels, channels, large_kernel_size, padding=large_kernel_size // 2, groups=channels, bias=False
        )
        self.large_normalization: nn.BatchNorm1d = nn.BatchNorm1d(channels)
        self.small: nn.Conv1d = nn.Conv1d(
            channels, channels, small_kernel_size, padding=small_kernel_size // 2, groups=channels, bias=False
        )
        self.small_normalization: nn.BatchNorm1d = nn.BatchNorm1d(channels)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the convolutions.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, channels, length).

        Returns
        -------
        Tensor
            Features with the shape of ``values``.
        """
        large: Tensor = self.large_normalization(self.large(values))
        small: Tensor = self.small_normalization(self.small(values))
        return large + small

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
