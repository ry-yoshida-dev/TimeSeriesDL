import math
from typing import override

import torch
from torch import Tensor, nn

from ..discretization import Discretization
from ..gated_linear_output import GatedLinearOutput
from ..linear_recurrence import LinearRecurrence
from ..normal_plus_low_rank import NormalPlusLowRank
from ..state_space_layer import StateSpaceLayer


class S5StateSpaceLayer(StateSpaceLayer):
    """
    S5 layer (Smith et al., 2023): one multi-input multi-output state space model over all channels, with a complex
    diagonal state matrix initialized from blocks of the normal part of HiPPO-LegS, evaluated by a parallel scan.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the shared state; the state matrix has state_size / 2 conjugate pairs.
    num_blocks : int
        Number of HiPPO-LegS blocks the state matrix is initialized from.
    discretization : Discretization
        Conversion of the continuous dynamics to steps.
    min_step : float
        Smallest initial step size.
    max_step : float
        Largest initial step size.
    dropout : float
        Dropout probability of the output stage.

    Raises
    ------
    ValueError
        If num_blocks does not split state_size into blocks of positive even size, or the step sizes are not
        0 < min_step <= max_step.
    """

    def __init__(
        self,
        size: int,
        state_size: int,
        num_blocks: int,
        discretization: Discretization,
        min_step: float,
        max_step: float,
        dropout: float,
    ) -> None:
        if num_blocks <= 0 or state_size % num_blocks != 0 or (state_size // num_blocks) % 2 != 0:
            raise ValueError(f"num_blocks {num_blocks} must split state_size {state_size} into blocks of even size")
        if not 0.0 < min_step <= max_step:
            raise ValueError(f"steps must satisfy 0 < min_step <= max_step, got {min_step} and {max_step}")
        super().__init__()
        self.discretization: Discretization = discretization
        block_size: int = state_size // num_blocks
        decomposition: NormalPlusLowRank = NormalPlusLowRank.from_legs(block_size)
        # Keep the eigenvalues with positive imaginary parts; their conjugates are implied.
        kept: slice = slice(block_size // 2, block_size)
        eigenvalues: Tensor = decomposition.eigenvalues[kept].repeat(num_blocks)
        eigenvectors: Tensor = decomposition.eigenvectors[:, kept]
        input_matrix: Tensor = torch.randn(num_blocks, block_size, size, dtype=torch.float64) / math.sqrt(size)
        output_matrix: Tensor = torch.randn(num_blocks, size, block_size, dtype=torch.complex128) / math.sqrt(
            state_size
        )
        projected_input: Tensor = torch.einsum("np,jnh->jph", eigenvectors.conj(), input_matrix.to(torch.complex128))
        projected_output: Tensor = torch.einsum("jhn,np->jhp", output_matrix, eigenvectors)
        self.log_negative_real: nn.Parameter = nn.Parameter(torch.log(-eigenvalues.real).float())
        self.imaginary: nn.Parameter = nn.Parameter(eigenvalues.imag.float())
        self.input_matrix: nn.Parameter = nn.Parameter(
            torch.view_as_real(projected_input.reshape(-1, size).to(torch.complex64)).clone()
        )
        self.output_matrix: nn.Parameter = nn.Parameter(
            torch.view_as_real(projected_output.permute(1, 0, 2).reshape(size, -1).to(torch.complex64)).clone()
        )
        self.skip: nn.Parameter = nn.Parameter(torch.randn(size))
        self.log_step: nn.Parameter = nn.Parameter(
            torch.rand(state_size // 2) * (math.log(max_step) - math.log(min_step)) + math.log(min_step)
        )
        self.output: GatedLinearOutput = GatedLinearOutput(size, dropout)

    @override
    def forward(self, values: Tensor) -> Tensor:
        eigenvalues: Tensor = torch.complex(-torch.exp(self.log_negative_real), self.imaginary)
        decay, input_scale = self.discretization.discretize_diagonal(eigenvalues, torch.exp(self.log_step))
        inputs: Tensor = torch.einsum("ph,blh->blp", torch.view_as_complex(self.input_matrix), values.to(decay.dtype))
        states: Tensor = LinearRecurrence.scan(decay, inputs * input_scale)
        outputs: Tensor = 2.0 * torch.einsum("hp,blp->blh", torch.view_as_complex(self.output_matrix), states).real
        return self.output(outputs + values * self.skip)
