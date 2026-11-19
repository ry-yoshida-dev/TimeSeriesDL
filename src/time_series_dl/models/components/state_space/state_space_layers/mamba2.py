import math
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..causal_depthwise_convolution import CausalDepthwiseConvolution
from ..state_space_layer import StateSpaceLayer


class Mamba2StateSpaceLayer(StateSpaceLayer):
    """
    Mamba-2 block (Dao and Gu, 2024): a selective state space model whose state matrix is a scalar per head, computed
    by the chunked state space duality algorithm, with input and output matrices shared by every head, a gated
    RMS normalization and parallel input projections.

    Parameters
    ----------
    size : int
        Number of channels.
    state_size : int
        Size of the state of each inner channel.
    expansion : int
        Ratio of the inner size to the size.
    head_size : int
        Number of inner channels per head; must divide the inner size.
    convolution_size : int
        Kernel size of the causal depthwise convolution.
    chunk_length : int
        Number of steps of each chunk; within a chunk the model is evaluated in its quadratic form.

    Raises
    ------
    ValueError
        If a size is not positive or head_size does not divide the inner size.
    """

    def __init__(
        self, size: int, state_size: int, expansion: int, head_size: int, convolution_size: int, chunk_length: int
    ) -> None:
        for name, value in (
            ("size", size),
            ("state_size", state_size),
            ("expansion", expansion),
            ("head_size", head_size),
            ("convolution_size", convolution_size),
            ("chunk_length", chunk_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        inner_size: int = expansion * size
        if inner_size % head_size != 0:
            raise ValueError(f"head_size {head_size} must divide the inner size {inner_size}")
        super().__init__()
        self.inner_size: int = inner_size
        self.state_size: int = state_size
        self.head_size: int = head_size
        self.num_heads: int = inner_size // head_size
        self.chunk_length: int = chunk_length
        convolved_size: int = inner_size + 2 * state_size
        self.input_projection: nn.Linear = nn.Linear(size, inner_size + convolved_size + self.num_heads, bias=False)
        self.convolution: CausalDepthwiseConvolution = CausalDepthwiseConvolution(convolved_size, convolution_size)
        steps: Tensor = torch.exp(torch.rand(self.num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.step_bias: nn.Parameter = nn.Parameter(steps + torch.log(-torch.expm1(-steps)))
        transitions: Tensor = torch.empty(self.num_heads).uniform_(1.0, 16.0)
        self.log_negative_transition: nn.Parameter = nn.Parameter(torch.log(transitions))
        self.skip: nn.Parameter = nn.Parameter(torch.ones(self.num_heads))
        self.normalization: nn.RMSNorm = nn.RMSNorm(inner_size)
        self.output_projection: nn.Linear = nn.Linear(inner_size, size, bias=False)

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch_size, length, _ = values.shape
        gate, convolved_input, step_input = self.input_projection(values).split(
            [self.inner_size, self.inner_size + 2 * self.state_size, self.num_heads], dim=-1
        )
        activated: Tensor = functional.silu(self.convolution(convolved_input))
        branch: Tensor = activated[..., : self.inner_size]
        input_matrix: Tensor = activated[..., self.inner_size : self.inner_size + self.state_size]
        output_matrix: Tensor = activated[..., self.inner_size + self.state_size :]
        steps: Tensor = functional.softplus(step_input + self.step_bias)
        heads: Tensor = branch.reshape(batch_size, length, self.num_heads, self.head_size)
        log_decays: Tensor = -torch.exp(self.log_negative_transition) * steps
        outputs: Tensor = self._scan_chunks(heads * steps[..., None], log_decays, input_matrix, output_matrix)
        outputs = outputs + heads * self.skip[:, None]
        normalized: Tensor = self.normalization(outputs.reshape(batch_size, length, -1) * functional.silu(gate))
        projected: Tensor = self.output_projection(normalized)
        return projected

    def _scan_chunks(self, inputs: Tensor, log_decays: Tensor, input_matrix: Tensor, output_matrix: Tensor) -> Tensor:
        # inputs (b, t, h, p), log_decays (b, t, h), input_matrix and output_matrix (b, t, n).
        length: int = inputs.shape[1]
        padding: int = -length % self.chunk_length
        num_chunks: int = (length + padding) // self.chunk_length

        def chunked(tensor: Tensor) -> Tensor:
            padded: Tensor = functional.pad(tensor.movedim(1, -1), (0, padding)).movedim(-1, 1)
            return padded.reshape(padded.shape[0], num_chunks, self.chunk_length, *padded.shape[2:])

        chunk_inputs: Tensor = chunked(inputs)
        chunk_input_matrix: Tensor = chunked(input_matrix)
        chunk_output_matrix: Tensor = chunked(output_matrix)
        decays: Tensor = chunked(log_decays).permute(0, 3, 1, 2)
        cumulative: Tensor = decays.cumsum(dim=-1)
        # Within each chunk: the quadratic form, attention-like with decay weights.
        within: Tensor = torch.einsum(
            "bcln,bcsn,bhcls,bcshp->bclhp",
            chunk_output_matrix,
            chunk_input_matrix,
            torch.exp(self._segment_sums(decays)),
            chunk_inputs,
        )
        # Across chunks: the state at the end of each chunk, carried by a recurrence over chunks.
        to_end: Tensor = torch.exp(cumulative[..., -1:] - cumulative)
        chunk_states: Tensor = torch.einsum("bcln,bhcl,bclhp->bchpn", chunk_input_matrix, to_end, chunk_inputs)
        chunk_states = torch.cat([torch.zeros_like(chunk_states[:, :1]), chunk_states], dim=1)
        chunk_decays: Tensor = torch.exp(self._segment_sums(functional.pad(cumulative[..., -1], (1, 0))))
        carried: Tensor = torch.einsum("bhzc,bchpn->bzhpn", chunk_decays, chunk_states)[:, :-1]
        across: Tensor = torch.einsum("bcln,bchpn,bhcl->bclhp", chunk_output_matrix, carried, torch.exp(cumulative))
        outputs: Tensor = (within + across).reshape(inputs.shape[0], -1, *inputs.shape[2:])
        return outputs[:, :length]

    @staticmethod
    def _segment_sums(values: Tensor) -> Tensor:
        # Entry (i, j) is the sum of values[j + 1 .. i] for j <= i, and -inf above the diagonal.
        length: int = values.shape[-1]
        repeated: Tensor = values[..., None].expand(*values.shape, length)
        strictly_lower: Tensor = torch.tril(torch.ones(length, length, dtype=torch.bool, device=values.device), -1)
        sums: Tensor = repeated.masked_fill(~strictly_lower, 0.0).cumsum(dim=-2)
        lower: Tensor = torch.tril(torch.ones(length, length, dtype=torch.bool, device=values.device))
        return sums.masked_fill(~lower, -torch.inf)
