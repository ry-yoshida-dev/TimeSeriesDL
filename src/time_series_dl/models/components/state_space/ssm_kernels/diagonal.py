import math
from typing import override

import torch
from torch import Tensor, nn

from ..diagonal_initialization import DiagonalInitialization
from ..discretization import Discretization
from ..ssm_kernel import SSMKernel


class DiagonalSSMKernel(SSMKernel):
    """
    Kernel of S4D (Gu et al., 2022): a complex diagonal state matrix in conjugate pairs, with the input matrix fixed
    to ones and a learned output matrix, computed as a Vandermonde product.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the state of each channel; the state matrix has state_size / 2 conjugate pairs.
    initialization : DiagonalInitialization
        Initial eigenvalues.
    discretization : Discretization
        Conversion of the continuous dynamics to steps.
    min_step : float
        Smallest initial step size.
    max_step : float
        Largest initial step size.

    Raises
    ------
    ValueError
        If state_size is not a positive even number.
    """

    def __init__(
        self,
        size: int,
        state_size: int,
        initialization: DiagonalInitialization,
        discretization: Discretization,
        min_step: float,
        max_step: float,
    ) -> None:
        if state_size <= 0 or state_size % 2 != 0:
            raise ValueError(f"state_size must be a positive even number, got {state_size}")
        super().__init__(size, min_step, max_step)
        self.discretization: Discretization = discretization
        eigenvalues: Tensor = initialization.eigenvalues(state_size // 2).expand(size, -1)
        self.log_negative_real: nn.Parameter = nn.Parameter(torch.log(-eigenvalues.real).clone())
        self.imaginary: nn.Parameter = nn.Parameter(eigenvalues.imag.clone())
        self.output_matrix: nn.Parameter = nn.Parameter(torch.randn(size, state_size // 2, 2) * math.sqrt(0.5))

    @override
    def forward(self, length: int) -> Tensor:
        eigenvalues: Tensor = torch.complex(-torch.exp(self.log_negative_real), self.imaginary)
        decay, input_scale = self.discretization.discretize_diagonal(eigenvalues, torch.exp(self.log_step)[:, None])
        weights: Tensor = torch.view_as_complex(self.output_matrix) * input_scale
        lags: Tensor = torch.arange(length, device=eigenvalues.device)
        powers: Tensor = torch.exp(torch.log(decay)[..., None] * lags)
        # Each pair and its conjugate contribute twice the real part.
        return 2.0 * torch.einsum("hn,hnl->hl", weights, powers).real
