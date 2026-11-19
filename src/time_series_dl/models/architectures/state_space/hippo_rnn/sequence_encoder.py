from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.state_space import HiPPOMemory
from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import HiPPORNNParameters


class HiPPORNNSequenceEncoder(SequenceEncoder):
    """
    Gated recurrent cell that reads the input and a HiPPO memory, and writes a learned scalar of its hidden state into
    the memory at every step; the hidden states are the features.

    Parameters
    ----------
    parameters : HiPPORNNParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: HiPPORNNParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.hidden_size
        self.memory: HiPPOMemory = HiPPOMemory(
            parameters.measure,
            parameters.state_size,
            input_length,
            parameters.window_ratio * input_length,
            parameters.discretization,
        )
        self.cell: nn.GRUCell = nn.GRUCell(input_channels + parameters.state_size, parameters.hidden_size)
        self.memory_input: nn.Linear = nn.Linear(parameters.hidden_size, 1)

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
            hidden = self.cell(torch.cat([values[:, index], memory], dim=-1), hidden)
            memory = self.memory.step(memory, self.memory_input(hidden).squeeze(-1), index)
            hidden_states.append(hidden)
        return torch.stack(hidden_states, dim=1)
