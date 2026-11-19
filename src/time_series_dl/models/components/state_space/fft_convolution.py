from typing import cast

import torch
from torch import Tensor


class FFTConvolution:
    """
    Causal convolution with a kernel as long as the sequence, computed with the fast Fourier transform.
    """

    @staticmethod
    def apply(values: Tensor, kernel: Tensor) -> Tensor:
        """
        Convolve causally along the last dimension: y_t = sum_k kernel_k values_{t-k}.

        Parameters
        ----------
        values : Tensor
            Real values with shape (..., length).
        kernel : Tensor
            Real kernel with shape (..., kernel_length), broadcastable with ``values`` except in the last dimension.

        Returns
        -------
        Tensor
            The convolved values with shape (..., length).
        """
        length: int = values.shape[-1]
        size: int = length + kernel.shape[-1]
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(values, n=size)) * cast(Tensor, torch.fft.rfft(kernel, n=size))
        convolved: Tensor = cast(Tensor, torch.fft.irfft(spectrum, n=size))[..., :length]
        return convolved
