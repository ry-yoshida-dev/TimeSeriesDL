from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class FreTSFrequencyLayer(nn.Module):
    """
    Frequency-domain MLP layer of FreTS: the embedded values are transformed by a real FFT along one axis, each
    frequency's embedding vector goes through a complex linear layer whose real and imaginary parts are activated by a
    ReLU and sparsified by a soft shrinkage, and the result is transformed back.

    Parameters
    ----------
    embed_size : int
        Size of each embedding vector.
    sparsity_threshold : float
        Threshold of the soft shrinkage of the real and imaginary parts.
    scale : float
        Standard deviation of the initial weights and biases.

    Raises
    ------
    ValueError
        If embed_size or scale is not positive, or sparsity_threshold is negative.
    """

    def __init__(self, embed_size: int, sparsity_threshold: float, scale: float) -> None:
        if embed_size <= 0:
            raise ValueError(f"embed_size must be positive, got {embed_size}")
        if sparsity_threshold < 0.0:
            raise ValueError(f"sparsity_threshold must not be negative, got {sparsity_threshold}")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        super().__init__()
        self.sparsity_threshold: float = sparsity_threshold
        self.weight: nn.Parameter = nn.Parameter(scale * torch.randn(embed_size, embed_size, 2))
        self.bias: nn.Parameter = nn.Parameter(scale * torch.randn(embed_size, 2))

    def forward(self, tokens: Tensor, dim: int) -> Tensor:
        """
        Filter the embedded values along one axis.

        Parameters
        ----------
        tokens : Tensor
            Embedded values whose last dimension is ``embed_size``.
        dim : int
            The axis transformed to the frequency domain; not the last one.

        Returns
        -------
        Tensor
            Filtered values with the shape of ``tokens``.
        """
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(tokens, dim=dim, norm="ortho"))
        weight_real: Tensor = self.weight[..., 0]
        weight_imaginary: Tensor = self.weight[..., 1]
        real: Tensor = functional.relu(spectrum.real @ weight_real - spectrum.imag @ weight_imaginary + self.bias[:, 0])
        imaginary: Tensor = functional.relu(
            spectrum.imag @ weight_real + spectrum.real @ weight_imaginary + self.bias[:, 1]
        )
        shrunk: Tensor = functional.softshrink(torch.stack([real, imaginary], dim=-1), self.sparsity_threshold)
        filtered: Tensor = cast(
            Tensor,
            torch.fft.irfft(torch.view_as_complex(shrunk), n=tokens.shape[dim], dim=dim, norm="ortho"),
        )
        return filtered

    def __call__(self, tokens: Tensor, dim: int) -> Tensor:
        return cast(Tensor, super().__call__(tokens, dim))
