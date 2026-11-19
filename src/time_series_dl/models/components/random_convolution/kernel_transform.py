from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class KernelTransform(nn.Module, ABC):
    """
    Fixed convolution kernels whose pooled outputs are the features of a linear model, as in the ROCKET family.

    Nothing in a transform is trained: its kernels and biases are buffers.
    """

    @staticmethod
    def convolve(series: Tensor, weights: Tensor, dilation: int) -> Tensor:
        """
        Convolve a batch of series with dilated kernels of odd length, padded with zeros to keep the length.

        The convolution is a matrix product with the gathered input windows, which on CPU is much faster than the
        NNPACK convolution PyTorch picks for many kernels without dilation.

        Parameters
        ----------
        series : Tensor
            Values with shape (batch, channels, length).
        weights : Tensor
            Kernels with shape (kernels, channels, width), width odd.
        dilation : int
            Spacing between the taps of the kernels.

        Returns
        -------
        Tensor
            Outputs with shape (batch, kernels, length).

        Raises
        ------
        ValueError
            If the width is even or the dilation is not positive.
        """
        width: int = weights.shape[-1]
        if width % 2 == 0 or dilation <= 0:
            raise ValueError(f"width must be odd and dilation positive, got {width} and {dilation}")
        length: int = series.shape[-1]
        padding: int = (width - 1) // 2 * dilation
        padded: Tensor = functional.pad(series, (padding, padding))
        taps: Tensor = torch.arange(length, device=series.device)[:, None] + dilation * torch.arange(
            width, device=series.device
        )
        windows: Tensor = padded[:, :, taps]
        return torch.einsum("bclw,kcw->bkl", windows, weights)

    @property
    @abstractmethod
    def num_features(self) -> int:
        """
        Get the number of features per sequence.

        Returns
        -------
        int
            The size of the feature vector.
        """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Transform a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Features with shape (batch, num_features).
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
