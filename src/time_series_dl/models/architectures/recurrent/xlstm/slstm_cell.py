from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional


class SLSTMCell(nn.Module):
    """
    sLSTM cell (Beck et al., 2024): a scalar memory with exponential input gate and sigmoid forget gate, a normalizer
    state and a stabilizer state, whose recurrent weights are block-diagonal over heads.

    The cell is inherently recurrent (its gates read the previous hidden state), so it runs step by step. Every state
    starts at zero. The forget gate biases follow the block-dependent power law initialization of the official code.

    Parameters
    ----------
    size : int
        Size of the hidden state.
    num_heads : int
        Number of heads; it must divide size.
    block_index : int
        Position of the block in the stack, for the forget gate initialization.
    num_blocks : int
        Number of blocks of the stack.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide size, or block_index is not in [0, num_blocks).
    """

    def __init__(self, size: int, num_heads: int, block_index: int, num_blocks: int) -> None:
        if size <= 0 or num_heads <= 0:
            raise ValueError(f"size and num_heads must be positive, got {size} and {num_heads}")
        if size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide size {size}")
        if not 0 <= block_index < num_blocks:
            raise ValueError(f"block_index must be in [0, {num_blocks}), got {block_index}")
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = size // num_heads
        # Recurrent weights of the input, forget, cell and output gates per head, zero at start as in the official
        # code.
        self.recurrent_weight: nn.Parameter = nn.Parameter(torch.zeros(num_heads, self.head_size, 4, self.head_size))
        ratio: float = block_index / (num_blocks - 1) if num_blocks > 1 else 0.0
        positions: Tensor = torch.arange(self.head_size) / max(self.head_size - 1, 1)
        forget_bias: Tensor = -(-5.0 + 12.0 * positions ** (0.3 + 1.3 * ratio))
        bias: Tensor = torch.zeros(4, num_heads, self.head_size)
        bias[1] = forget_bias
        self.bias: nn.Parameter = nn.Parameter(bias)

    def forward(self, input_gates: Tensor, forget_gates: Tensor, cell_inputs: Tensor, output_gates: Tensor) -> Tensor:
        """
        Run the cell over a sequence.

        Parameters
        ----------
        input_gates : Tensor
            Input contributions to the input gate pre-activations with shape (batch, length, size).
        forget_gates : Tensor
            Input contributions to the forget gate pre-activations with shape (batch, length, size).
        cell_inputs : Tensor
            Input contributions to the cell input pre-activations with shape (batch, length, size).
        output_gates : Tensor
            Input contributions to the output gate pre-activations with shape (batch, length, size).

        Returns
        -------
        Tensor
            Hidden states with shape (batch, length, size).
        """
        contributions: Tensor = torch.stack([input_gates, forget_gates, cell_inputs, output_gates], dim=2)
        contributions = contributions.reshape(*contributions.shape[:-1], self.num_heads, self.head_size) + self.bias
        batch_size: int = contributions.shape[0]
        hidden: Tensor = contributions.new_zeros(batch_size, self.num_heads, self.head_size)
        cell: Tensor = torch.zeros_like(hidden)
        normalizer: Tensor = torch.zeros_like(hidden)
        stabilizer: Tensor = torch.zeros_like(hidden)
        hidden_states: list[Tensor] = []
        for step in range(contributions.shape[1]):
            recurrent: Tensor = torch.einsum("bhd,hdge->bghe", hidden, self.recurrent_weight)
            input_preactivation, forget_preactivation, cell_preactivation, output_preactivation = (
                contributions[:, step] + recurrent
            ).unbind(dim=1)
            log_forget: Tensor = stabilizer + functional.logsigmoid(forget_preactivation)
            stabilizer = torch.maximum(input_preactivation, log_forget)
            input_gate: Tensor = torch.exp(input_preactivation - stabilizer)
            forget_gate: Tensor = torch.exp(log_forget - stabilizer)
            cell = forget_gate * cell + input_gate * torch.tanh(cell_preactivation)
            normalizer = forget_gate * normalizer + input_gate
            hidden = torch.sigmoid(output_preactivation) * cell / normalizer
            hidden_states.append(hidden)
        return torch.stack(hidden_states, dim=1).flatten(start_dim=-2)

    def __call__(self, input_gates: Tensor, forget_gates: Tensor, cell_inputs: Tensor, output_gates: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(input_gates, forget_gates, cell_inputs, output_gates))
