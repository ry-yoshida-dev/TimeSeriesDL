import math
from typing import override

import torch
from torch import Tensor, nn

from ..gated_linear_output import GatedLinearOutput
from ..linear_recurrence import LinearRecurrence
from ..state_space_layer import StateSpaceLayer


class LRUStateSpaceLayer(StateSpaceLayer):
    """
    Linear Recurrent Unit (Orvieto et al., 2023): a complex diagonal recurrence parametrized directly in discrete time
    by the log of its modulus and phase, with input normalization, evaluated by a parallel scan.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the complex state.
    min_radius : float
        Smallest initial modulus of the eigenvalues.
    max_radius : float
        Largest initial modulus of the eigenvalues; moduli are sampled uniformly on the ring between the two.
    max_phase : float
        Largest initial phase of the eigenvalues.
    dropout : float
        Dropout probability of the output stage.

    Raises
    ------
    ValueError
        If state_size is not positive, the radii are not 0 < min_radius <= max_radius < 1, or max_phase is not
        positive.
    """

    def __init__(
        self, size: int, state_size: int, min_radius: float, max_radius: float, max_phase: float, dropout: float
    ) -> None:
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if not 0.0 < min_radius <= max_radius < 1.0:
            raise ValueError(f"radii must satisfy 0 < min_radius <= max_radius < 1, got {min_radius}, {max_radius}")
        if max_phase <= 0.0:
            raise ValueError(f"max_phase must be positive, got {max_phase}")
        super().__init__()
        squared_radius: Tensor = torch.rand(state_size) * (max_radius**2 - min_radius**2) + min_radius**2
        self.log_rate: nn.Parameter = nn.Parameter(torch.log(-0.5 * torch.log(squared_radius)))
        self.log_phase: nn.Parameter = nn.Parameter(torch.log(max_phase * torch.rand(state_size).clamp_min(1e-4)))
        self.log_input_norm: nn.Parameter = nn.Parameter(0.5 * torch.log(1.0 - squared_radius))
        self.input_matrix: nn.Parameter = nn.Parameter(torch.randn(state_size, size, 2) / math.sqrt(2 * size))
        self.output_matrix: nn.Parameter = nn.Parameter(torch.randn(size, state_size, 2) / math.sqrt(state_size))
        self.skip: nn.Parameter = nn.Parameter(torch.randn(size))
        self.output: GatedLinearOutput = GatedLinearOutput(size, dropout)

    @override
    def forward(self, values: Tensor) -> Tensor:
        decay: Tensor = torch.exp(torch.complex(-torch.exp(self.log_rate), torch.exp(self.log_phase)))
        input_matrix: Tensor = torch.view_as_complex(self.input_matrix) * torch.exp(self.log_input_norm)[:, None]
        inputs: Tensor = torch.einsum("nh,blh->bln", input_matrix, values.to(decay.dtype))
        states: Tensor = LinearRecurrence.scan(decay, inputs)
        outputs: Tensor = torch.einsum("hn,bln->blh", torch.view_as_complex(self.output_matrix), states).real
        return self.output(outputs + values * self.skip)
