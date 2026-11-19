from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .log_space_scan import LogSpaceScan
from .minimal_recurrent_cell import MinimalRecurrentCell


class MinimalRecurrentLayer(nn.Module):
    """
    minGRU or minLSTM layer (Feng et al., 2024) followed by a linear output map.

    The gates and the candidate state are linear maps of the input alone, so the hidden states follow a linear
    recurrence that ``forward`` evaluates with a parallel scan in log space. As in the paper's log-space version,
    the candidate state passes through g(x) = x + 0.5 for x >= 0 and sigmoid(x) otherwise, which keeps it positive.
    ``recurrent`` evaluates the same layer step by step. The initial state is zero.

    Parameters
    ----------
    size : int
        Size of the input and output vectors.
    hidden_size : int
        Size of the hidden state.
    cell : MinimalRecurrentCell
        Gating of the recurrence.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, size: int, hidden_size: int, cell: MinimalRecurrentCell) -> None:
        if size <= 0 or hidden_size <= 0:
            raise ValueError(f"size and hidden_size must be positive, got {size} and {hidden_size}")
        super().__init__()
        self.cell: MinimalRecurrentCell = cell
        self.input_projection: nn.Linear = nn.Linear(size, cell.num_projections * hidden_size)
        self.output_projection: nn.Linear = nn.Linear(hidden_size, size)

    @staticmethod
    def _positive(values: Tensor) -> Tensor:
        return torch.where(values >= 0, values + 0.5, torch.sigmoid(values))

    @staticmethod
    def _log_positive(values: Tensor) -> Tensor:
        return torch.where(values >= 0, torch.log(functional.relu(values) + 0.5), -functional.softplus(-values))

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the layer with the parallel scan.

        Parameters
        ----------
        values : Tensor
            Inputs with shape (batch, length, size).

        Returns
        -------
        Tensor
            Outputs with the shape of ``values``.
        """
        projections: tuple[Tensor, ...] = self.input_projection(values).chunk(self.cell.num_projections, dim=-1)
        log_candidates: Tensor = self._log_positive(projections[-1])
        match self.cell:
            case MinimalRecurrentCell.GRU:
                log_coefficients: Tensor = -functional.softplus(projections[0])
                log_inputs: Tensor = -functional.softplus(-projections[0])
            case MinimalRecurrentCell.LSTM:
                # log f' and log i' of the normalized gates f / (f + i) and i / (f + i).
                difference: Tensor = functional.softplus(-projections[0]) - functional.softplus(-projections[1])
                log_coefficients = -functional.softplus(difference)
                log_inputs = -functional.softplus(-difference)
        hidden: Tensor = LogSpaceScan.scan(log_coefficients, log_inputs + log_candidates)
        outputs: Tensor = self.output_projection(hidden)
        return outputs

    def recurrent(self, values: Tensor) -> Tensor:
        """
        Apply the layer step by step, as a reference for the parallel scan.

        Parameters
        ----------
        values : Tensor
            Inputs with shape (batch, length, size).

        Returns
        -------
        Tensor
            Outputs with the shape of ``values``.
        """
        projections: tuple[Tensor, ...] = self.input_projection(values).chunk(self.cell.num_projections, dim=-1)
        candidates: Tensor = self._positive(projections[-1])
        match self.cell:
            case MinimalRecurrentCell.GRU:
                inputs: Tensor = torch.sigmoid(projections[0])
                coefficients: Tensor = 1.0 - inputs
            case MinimalRecurrentCell.LSTM:
                forget: Tensor = torch.sigmoid(projections[0])
                input_gate: Tensor = torch.sigmoid(projections[1])
                coefficients = forget / (forget + input_gate)
                inputs = input_gate / (forget + input_gate)
        hidden: Tensor = torch.zeros_like(candidates[:, 0])
        hidden_states: list[Tensor] = []
        for step in range(values.shape[1]):
            hidden = coefficients[:, step] * hidden + inputs[:, step] * candidates[:, step]
            hidden_states.append(hidden)
        outputs: Tensor = self.output_projection(torch.stack(hidden_states, dim=1))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
