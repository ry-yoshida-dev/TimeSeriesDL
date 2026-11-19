from typing import override

import torch
from torch import Tensor, nn

from ....components.differential_equations import CubicInterpolation, CubicPath, ODESolver
from ....interfaces.irregular_sequence_encoder import IrregularSequenceEncoder
from ....interfaces.irregular_tensors import IrregularTensors
from .vector_field import NeuralCDEVectorField


class NeuralCDESequenceEncoder(IrregularSequenceEncoder):
    """
    Neural controlled differential equation (Kidger et al., 2020) over a cubic control path of the observations.

    The path has the time, the values filled forward in each channel (zero before a channel's first observation),
    and the observational intensity of each channel (its cumulative number of observations). The initial hidden
    state is a linear map of the path's first knot, and the equation is solved knot to knot with a fixed number of
    steps per interval; the final hidden state at each sample's last observation is the feature vector. The equation
    does not depend on how time is parametrized, so each interval is solved over the position in [0, 1] within it.

    Parameters
    ----------
    input_channels : int
        Number of channels of the sequences.
    hidden_size : int
        Size of the hidden state.
    field_hidden_size : int
        Size of the hidden layers of the vector field.
    field_layers : int
        Number of hidden layers of the vector field.
    interpolation : CubicInterpolation
        Interpolation of the control path.
    solver : ODESolver
        Fixed-step solver.
    steps_per_interval : int
        Number of solver steps between consecutive observations.

    Raises
    ------
    ValueError
        If a size or steps_per_interval is not positive.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        field_hidden_size: int,
        field_layers: int,
        interpolation: CubicInterpolation,
        solver: ODESolver,
        steps_per_interval: int,
    ) -> None:
        if steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {steps_per_interval}")
        super().__init__()
        path_channels: int = 1 + 2 * input_channels
        self.hidden_size: int = hidden_size
        self.interpolation: CubicInterpolation = interpolation
        self.solver: ODESolver = solver
        self.steps_per_interval: int = steps_per_interval
        self.initial: nn.Linear = nn.Linear(path_channels, hidden_size)
        self.vector_field: NeuralCDEVectorField = NeuralCDEVectorField(
            hidden_size, path_channels, field_hidden_size, field_layers
        )

    @property
    @override
    def output_size(self) -> int:
        return self.hidden_size

    @override
    def forward(self, sequences: IrregularTensors) -> Tensor:
        path: CubicPath = self.interpolation.fit(
            sequences.times, self.build_knots(sequences), sequences.lengths.clamp(min=1)
        )
        state: Tensor = self.initial(path.constant[:, 0])
        start: Tensor = torch.zeros_like(state[:, :1])
        end: Tensor = torch.ones_like(start)
        for interval in range(path.num_intervals):

            def field(position: Tensor, hidden: Tensor, interval: int = interval) -> Tensor:
                return self.vector_field(hidden, path.derivative(interval, position))

            state = self.solver.integrate(field, state, start, end, self.steps_per_interval)
        return state

    @staticmethod
    def build_knots(sequences: IrregularTensors) -> Tensor:
        """
        Build the knots of the control path: time, filled-forward values and observational intensity.

        Parameters
        ----------
        sequences : IrregularTensors
            The padded sequences.

        Returns
        -------
        Tensor
            The knots with shape (batch, length, 1 + 2 * channels); padding steps repeat the last knot.
        """
        mask: Tensor = sequences.observed_mask
        steps: Tensor = torch.arange(mask.shape[1], device=mask.device).view(1, -1, 1)
        last_observed: Tensor = torch.where(mask > 0, steps, torch.full_like(steps, -1)).cummax(dim=1).values
        filled: Tensor = sequences.values.gather(1, last_observed.clamp(min=0))
        filled = torch.where(last_observed >= 0, filled, torch.zeros_like(filled))
        return torch.cat([sequences.times.unsqueeze(-1), filled, mask.cumsum(dim=1)], dim=-1)
