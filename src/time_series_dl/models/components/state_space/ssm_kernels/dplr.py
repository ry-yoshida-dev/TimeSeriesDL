import math
from typing import cast, override

import torch
from torch import Tensor, nn

from ..normal_plus_low_rank import NormalPlusLowRank
from ..ssm_kernel import SSMKernel


class DPLRSSMKernel(SSMKernel):
    """
    Kernel of S4 (Gu et al., 2022): a diagonal plus low-rank state matrix A = Lambda - P P*, initialized from
    HiPPO-LegS and discretized with the bilinear transform.

    The kernel is evaluated at the roots of unity through Cauchy products and the Woodbury identity, and brought back
    with an inverse FFT; the output matrix is learned in the form C (I - A_bar^L), as in the paper.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the complex state of each channel.
    min_step : float
        Smallest initial step size.
    max_step : float
        Largest initial step size.

    Raises
    ------
    ValueError
        If state_size is not positive.
    """

    def __init__(self, size: int, state_size: int, min_step: float, max_step: float) -> None:
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        super().__init__(size, min_step, max_step)
        decomposition: NormalPlusLowRank = NormalPlusLowRank.from_legs(state_size)
        eigenvalues: Tensor = decomposition.eigenvalues.to(torch.complex64).expand(size, -1)
        self.log_negative_real: nn.Parameter = nn.Parameter(torch.log(-eigenvalues.real).clone())
        self.imaginary: nn.Parameter = nn.Parameter(eigenvalues.imag.clone())
        self.low_rank: nn.Parameter = nn.Parameter(
            torch.view_as_real(decomposition.low_rank.to(torch.complex64)).expand(size, -1, -1).clone()
        )
        self.input_vector: nn.Parameter = nn.Parameter(
            torch.view_as_real(decomposition.input_vector.to(torch.complex64)).expand(size, -1, -1).clone()
        )
        self.output_matrix: nn.Parameter = nn.Parameter(torch.randn(size, state_size, 2) * math.sqrt(0.5))

    @override
    def forward(self, length: int) -> Tensor:
        # The roots of unity of an even length include -1, where the bilinear transform is singular; the learned
        # output matrix stands for any truncation length, so an odd length is evaluated and cut.
        evaluated_length: int = length if length % 2 == 1 else length + 1
        eigenvalues: Tensor = torch.complex(-torch.exp(self.log_negative_real), self.imaginary)
        low_rank: Tensor = torch.view_as_complex(self.low_rank)
        input_vector: Tensor = torch.view_as_complex(self.input_vector)
        output_matrix: Tensor = torch.view_as_complex(self.output_matrix)
        angles: Tensor = -2.0 * torch.pi * torch.arange(evaluated_length, device=eigenvalues.device) / evaluated_length
        roots: Tensor = torch.polar(torch.ones_like(angles), angles)
        frequencies: Tensor = (2.0 / torch.exp(self.log_step))[:, None] * ((1.0 - roots) / (1.0 + roots))
        denominators: Tensor = frequencies[:, :, None] - eigenvalues[:, None, :]
        output_input: Tensor = self._cauchy(output_matrix.conj() * input_vector, denominators)
        output_rank: Tensor = self._cauchy(output_matrix.conj() * low_rank, denominators)
        rank_input: Tensor = self._cauchy(low_rank.conj() * input_vector, denominators)
        rank_rank: Tensor = self._cauchy(low_rank.conj() * low_rank, denominators)
        at_roots: Tensor = (2.0 / (1.0 + roots)) * (output_input - output_rank * rank_input / (1.0 + rank_rank))
        kernel: Tensor = cast(Tensor, torch.fft.ifft(at_roots, n=evaluated_length)).real[:, :length]
        return kernel

    @staticmethod
    def _cauchy(numerators: Tensor, denominators: Tensor) -> Tensor:
        return (numerators[:, None, :] / denominators).sum(dim=-1)
