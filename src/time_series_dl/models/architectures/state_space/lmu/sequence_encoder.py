from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.state_space import HiPPOMeasure, HiPPOMemory
from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import LMUParameters


class LMUSequenceEncoder(SequenceEncoder):
    """
    Legendre Memory Unit cell: a scalar encoding of the input, hidden state and memory drives a Legendre memory of a
    sliding window, and the hidden state is a tanh layer of the input, hidden state and updated memory.

    Parameters
    ----------
    parameters : LMUParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: LMUParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.hidden_size
        self.memory: HiPPOMemory = HiPPOMemory(
            HiPPOMeasure.LEGT,
            parameters.state_size,
            input_length,
            parameters.window_ratio * input_length,
            parameters.discretization,
        )
        combined_size: int = input_channels + parameters.hidden_size + parameters.state_size
        self.memory_input: nn.Linear = nn.Linear(combined_size, 1, bias=False)
        self.hidden_layer: nn.Linear = nn.Linear(combined_size, parameters.hidden_size)

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        hidden: Tensor = values.new_zeros(values.shape[0], self._output_size)
        memory: Tensor = values.new_zeros(values.shape[0], self.memory.state_size)
        hidden_states: list[Tensor] = []
        for index in range(values.shape[1]):
            step_input: Tensor = values[:, index]
            encoded: Tensor = self.memory_input(torch.cat([step_input, hidden, memory], dim=-1)).squeeze(-1)
            memory = self.memory.step(memory, encoded, index)
            hidden = torch.tanh(self.hidden_layer(torch.cat([step_input, hidden, memory], dim=-1)))
            hidden_states.append(hidden)
        return torch.stack(hidden_states, dim=1)
