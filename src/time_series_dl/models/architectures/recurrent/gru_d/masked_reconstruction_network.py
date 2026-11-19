from typing import Final, override

import torch
from deep_learning_loss import MSELoss
from torch import Tensor, nn

from ....components.missing_values import LastObservation, TemporalDecay, TimeGap
from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction import MaskedReconstruction
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork

_SQUARED_ERROR: Final[MSELoss] = MSELoss()


class GRUDMaskedReconstructionNetwork(MaskedReconstructionNetwork):
    """
    GRU-D (Che et al., 2018): a GRU whose missing inputs decay from the last observation towards the channel mean,
    and whose hidden state decays with the time since the last observations.

    Each step's values are read from the decayed hidden state before the step's update, so every estimate uses only
    earlier steps. The network's own loss is the mean squared error of these estimates on the observed inputs; the
    task's loss on the hidden values is added with weight one.

    Parameters
    ----------
    hidden_size : int
        Size of the hidden state.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If hidden_size is not positive.
    """

    def __init__(self, hidden_size: int, shape: SequenceShape) -> None:
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        super().__init__()
        self.hidden_size: int = hidden_size
        # The empirical mean of each channel, learned; zero is the mean of the observed values after standard scaling.
        self.mean: nn.Parameter = nn.Parameter(torch.zeros(shape.channels))
        self.input_decay: TemporalDecay = TemporalDecay(shape.channels, shape.channels, is_diagonal=True)
        self.hidden_decay: TemporalDecay = TemporalDecay(shape.channels, hidden_size, is_diagonal=False)
        self.cell: nn.GRUCell = nn.GRUCell(2 * shape.channels, hidden_size)
        self.output_projection: nn.Linear = nn.Linear(hidden_size, shape.channels)

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return 1.0

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        mask: Tensor = is_observed.to(values.dtype)
        gaps: Tensor = TimeGap.compute(is_observed)
        input_decays: Tensor = self.input_decay(gaps)
        hidden_decays: Tensor = self.hidden_decay(gaps)
        last_observed: Tensor = LastObservation.carry_forward(values, is_observed, self.mean)
        decayed: Tensor = input_decays * last_observed + (1.0 - input_decays) * self.mean
        inputs: Tensor = torch.cat([mask * values + (1.0 - mask) * decayed, mask], dim=2)
        hidden: Tensor = values.new_zeros(values.shape[0], self.hidden_size)
        states: list[Tensor] = []
        for step in range(values.shape[1]):
            hidden = hidden * hidden_decays[:, step]
            states.append(hidden)
            hidden = self.cell(inputs[:, step], hidden)
        estimates: Tensor = self.output_projection(torch.stack(states, dim=1))
        return MaskedReconstruction(
            values=estimates, loss=_SQUARED_ERROR.compute_masked(estimates, values, is_observed)
        )
