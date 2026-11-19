import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .headwise_layer_norm import HeadwiseLayerNorm


class MLSTMCell(nn.Module):
    """
    mLSTM cell (Beck et al., 2024): a matrix memory per head updated with the outer product of each step's value and
    key, under an exponential input gate and a sigmoid forget gate, read out with the query and divided by a
    normalizer state.

    The gates of each head are computed from the concatenated queries, keys and values. ``forward`` evaluates the
    parallel form, which weights every pair of steps by its accumulated gates at once, and ``recurrent`` the
    equivalent step-by-step form with the stabilizer state of the paper. The outputs are normalized per head.

    Parameters
    ----------
    size : int
        Size of the queries, keys and values.
    num_heads : int
        Number of heads; it must divide size.

    Raises
    ------
    ValueError
        If a size is not positive or num_heads does not divide size.
    """

    EPSILON: float = 1e-6

    def __init__(self, size: int, num_heads: int) -> None:
        if size <= 0 or num_heads <= 0:
            raise ValueError(f"size and num_heads must be positive, got {size} and {num_heads}")
        if size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide size {size}")
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = size // num_heads
        self.input_gate: nn.Linear = nn.Linear(3 * size, num_heads)
        self.forget_gate: nn.Linear = nn.Linear(3 * size, num_heads)
        # Initialization of the official code: forget gates start mostly open, input gates near exp(0).
        with torch.no_grad():
            _ = self.input_gate.weight.zero_()
            _ = self.input_gate.bias.normal_(0.0, 0.1)
            _ = self.forget_gate.weight.zero_()
            _ = self.forget_gate.bias.copy_(torch.linspace(3.0, 6.0, num_heads))
        self.output_normalization: HeadwiseLayerNorm = HeadwiseLayerNorm(size, num_heads)

    def _split_heads(self, values: Tensor) -> Tensor:
        heads: Tensor = values.reshape(*values.shape[:-1], self.num_heads, self.head_size)
        return heads.transpose(1, 2)

    def _gates(self, queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        gate_inputs: Tensor = torch.cat([queries, keys, values], dim=-1)
        input_preactivations: Tensor = self.input_gate(gate_inputs).transpose(1, 2)
        log_forget_gates: Tensor = functional.logsigmoid(self.forget_gate(gate_inputs)).transpose(1, 2)
        return input_preactivations, log_forget_gates

    def _merge_heads(self, hidden: Tensor) -> Tensor:
        return self.output_normalization(hidden.transpose(1, 2).flatten(start_dim=-2))

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """
        Evaluate the cell in parallel over the sequence.

        Parameters
        ----------
        queries : Tensor
            Queries with shape (batch, length, size).
        keys : Tensor
            Keys with shape (batch, length, size).
        values : Tensor
            Values with shape (batch, length, size).

        Returns
        -------
        Tensor
            Normalized hidden states with shape (batch, length, size).
        """
        input_preactivations, log_forget_gates = self._gates(queries, keys, values)
        length: int = queries.shape[1]
        cumulative: Tensor = log_forget_gates.cumsum(dim=-1)
        # log_weights[t, s] = sum of the log forget gates of steps s+1..t, plus the input gate of step s.
        log_weights: Tensor = cumulative[..., :, None] - cumulative[..., None, :] + input_preactivations[..., None, :]
        is_future: Tensor = torch.ones(length, length, dtype=torch.bool, device=queries.device).triu(diagonal=1)
        log_weights = log_weights.masked_fill(is_future, -math.inf)
        stabilizer: Tensor = log_weights.amax(dim=-1, keepdim=True)
        weights: Tensor = torch.exp(log_weights - stabilizer)
        query_heads: Tensor = self._split_heads(queries)
        key_heads: Tensor = self._split_heads(keys) / math.sqrt(self.head_size)
        scores: Tensor = (query_heads @ key_heads.transpose(-2, -1)) * weights
        normalizer: Tensor = torch.maximum(scores.sum(dim=-1, keepdim=True).abs(), torch.exp(-stabilizer))
        hidden: Tensor = (scores / (normalizer + self.EPSILON)) @ self._split_heads(values)
        return self._merge_heads(hidden)

    def recurrent(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """
        Evaluate the cell step by step, as a reference for the parallel form.

        Parameters
        ----------
        queries : Tensor
            Queries with shape (batch, length, size).
        keys : Tensor
            Keys with shape (batch, length, size).
        values : Tensor
            Values with shape (batch, length, size).

        Returns
        -------
        Tensor
            Normalized hidden states with shape (batch, length, size).
        """
        input_preactivations, log_forget_gates = self._gates(queries, keys, values)
        query_heads: Tensor = self._split_heads(queries)
        key_heads: Tensor = self._split_heads(keys) / math.sqrt(self.head_size)
        value_heads: Tensor = self._split_heads(values)
        batch_size: int = queries.shape[0]
        memory: Tensor = queries.new_zeros(batch_size, self.num_heads, self.head_size, self.head_size)
        normalizer: Tensor = queries.new_zeros(batch_size, self.num_heads, self.head_size)
        stabilizer: Tensor = queries.new_zeros(batch_size, self.num_heads)
        hidden_states: list[Tensor] = []
        for step in range(queries.shape[1]):
            log_forget: Tensor = log_forget_gates[..., step] + stabilizer
            stabilizer = torch.maximum(log_forget, input_preactivations[..., step])
            forget: Tensor = torch.exp(log_forget - stabilizer)[..., None]
            input_gate: Tensor = torch.exp(input_preactivations[..., step] - stabilizer)[..., None]
            key: Tensor = key_heads[:, :, step]
            memory = forget[..., None] * memory + input_gate[..., None] * (
                value_heads[:, :, step, :, None] * key[..., None, :]
            )
            normalizer = forget * normalizer + input_gate * key
            query: Tensor = query_heads[:, :, step]
            numerator: Tensor = (memory @ query[..., None]).squeeze(-1)
            denominator: Tensor = torch.maximum(
                (normalizer * query).sum(dim=-1, keepdim=True).abs(), torch.exp(-stabilizer)[..., None]
            )
            hidden_states.append(numerator / (denominator + self.EPSILON))
        return self._merge_heads(torch.stack(hidden_states, dim=2))

    def __call__(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(queries, keys, values))
