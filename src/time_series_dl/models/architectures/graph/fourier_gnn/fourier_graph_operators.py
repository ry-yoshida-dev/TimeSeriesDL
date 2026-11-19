from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import FourierGNNParameters


class FourierGNNGraphOperators(nn.Module):
    """
    Stacked Fourier graph operators of FourierGNN, applied to the spectrum of the hypervariate graph.

    Each operator multiplies the spectrum by a complex weight matrix, adds a complex bias and applies a rectifier to
    the real and imaginary parts. The soft-shrunk outputs of all operators are summed and added to the input
    spectrum, which in the time domain amounts to summing graph convolutions of increasing order over the fully
    connected graph of all (channel, step) pairs.

    Parameters
    ----------
    parameters : FourierGNNParameters
        The model configuration.
    """

    INITIAL_SCALE: float = 0.02

    def __init__(self, parameters: FourierGNNParameters) -> None:
        super().__init__()
        size: int = parameters.embedding_size
        self.sparsity_threshold: float = parameters.sparsity_threshold
        # Index 0 of the second dimension holds the real parts, index 1 the imaginary parts.
        self.weights: nn.Parameter = nn.Parameter(
            self.INITIAL_SCALE * torch.randn(parameters.num_operators, 2, size, size)
        )
        self.biases: nn.Parameter = nn.Parameter(self.INITIAL_SCALE * torch.randn(parameters.num_operators, 2, size))

    def forward(self, spectrum: Tensor) -> Tensor:
        """
        Apply the operators.

        Parameters
        ----------
        spectrum : Tensor
            Complex spectrum with shape (batch, frequencies, embedding_size).

        Returns
        -------
        Tensor
            Complex spectrum of the same shape.
        """
        real: Tensor = spectrum.real
        imaginary: Tensor = spectrum.imag
        total_real: Tensor = torch.zeros_like(real)
        total_imaginary: Tensor = torch.zeros_like(imaginary)
        for weight, bias in zip(self.weights, self.biases, strict=True):
            next_real: Tensor = functional.relu(real @ weight[0] - imaginary @ weight[1] + bias[0])
            next_imaginary: Tensor = functional.relu(imaginary @ weight[0] + real @ weight[1] + bias[1])
            real, imaginary = next_real, next_imaginary
            total_real = total_real + functional.softshrink(real, self.sparsity_threshold)
            total_imaginary = total_imaginary + functional.softshrink(imaginary, self.sparsity_threshold)
        return torch.complex(total_real, total_imaginary) + spectrum

    def __call__(self, spectrum: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(spectrum))
