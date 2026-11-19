import math
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..causal_depthwise_convolution import CausalDepthwiseConvolution
from ..linear_recurrence import LinearRecurrence
from ..state_space_layer import StateSpaceLayer


class MambaStateSpaceLayer(StateSpaceLayer):
    """
    Mamba block (Gu and Dao, 2024): an expanded, convolved and gated branch through a selective state space model,
    whose step size and input and output matrices depend on the input at each step.

    The selective scan runs as a parallel scan in PyTorch instead of the fused CUDA kernel.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the state of each inner channel.
    expansion : int
        Ratio of the inner size to the size.
    convolution_size : int
        Kernel size of the causal depthwise convolution.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, size: int, state_size: int, expansion: int, convolution_size: int) -> None:
        for name, value in (
            ("size", size),
            ("state_size", state_size),
            ("expansion", expansion),
            ("convolution_size", convolution_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        super().__init__()
        inner_size: int = expansion * size
        self.step_rank: int = math.ceil(size / 16)
        self.state_size: int = state_size
        self.input_projection: nn.Linear = nn.Linear(size, 2 * inner_size, bias=False)
        self.convolution: CausalDepthwiseConvolution = CausalDepthwiseConvolution(inner_size, convolution_size)
        self.selection: nn.Linear = nn.Linear(inner_size, self.step_rank + 2 * state_size, bias=False)
        self.step_projection: nn.Linear = nn.Linear(self.step_rank, inner_size)
        _ = nn.init.uniform_(self.step_projection.weight, -(self.step_rank**-0.5), self.step_rank**-0.5)
        # Initial step sizes are log-uniform in [0.001, 0.1], through the inverse of the softplus.
        steps: Tensor = torch.exp(torch.rand(inner_size) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            _ = self.step_projection.bias.copy_(steps + torch.log(-torch.expm1(-steps)))
        self.log_negative_transition: nn.Parameter = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)).repeat(inner_size, 1)
        )
        self.skip: nn.Parameter = nn.Parameter(torch.ones(inner_size))
        self.output_projection: nn.Linear = nn.Linear(inner_size, size, bias=False)

    @override
    def forward(self, values: Tensor) -> Tensor:
        branch, gate = self.input_projection(values).chunk(2, dim=-1)
        activated: Tensor = functional.silu(self.convolution(branch))
        step_input, input_matrix, output_matrix = self.selection(activated).split(
            [self.step_rank, self.state_size, self.state_size], dim=-1
        )
        steps: Tensor = functional.softplus(self.step_projection(step_input))
        decays: Tensor = torch.exp(steps[..., None] * -torch.exp(self.log_negative_transition))
        inputs: Tensor = (steps * activated)[..., None] * input_matrix[:, :, None, :]
        states: Tensor = LinearRecurrence.scan(decays, inputs)
        outputs: Tensor = torch.einsum("blin,bln->bli", states, output_matrix) + activated * self.skip
        projected: Tensor = self.output_projection(outputs * functional.silu(gate))
        return projected
