from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from ....components.differential_equations import ODESolver
from ....components.layers import Activation, MultilayerPerceptron
from ....interfaces.irregular_tensors import IrregularTensors


class ODERNNRecurrence(nn.Module):
    """
    ODE-RNN (Rubanova et al., 2019): between observations the hidden state follows an ordinary differential
    equation, and at each observation a GRU cell updates it from the values and the observed mask.

    Read forward, the state starts at zero at time 0 and ends at each sample's last observation. Read backward, as
    the recognition network of the Latent ODE, it starts at zero at the last observation, reads the observations in
    reverse and ends at time 0.

    Parameters
    ----------
    input_channels : int
        Number of channels of the sequences.
    hidden_size : int
        Size of the hidden state.
    ode_hidden_sizes : Sequence[int]
        Hidden layer sizes of the tanh network that defines the derivative of the hidden state.
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
        ode_hidden_sizes: Sequence[int],
        solver: ODESolver,
        steps_per_interval: int,
    ) -> None:
        if steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {steps_per_interval}")
        super().__init__()
        self.hidden_size: int = hidden_size
        self.solver: ODESolver = solver
        self.steps_per_interval: int = steps_per_interval
        self.derivative: MultilayerPerceptron = MultilayerPerceptron(
            hidden_size, ode_hidden_sizes, hidden_size, Activation.TANH
        )
        self.cell: nn.GRUCell = nn.GRUCell(2 * input_channels, hidden_size)

    def forward(self, sequences: IrregularTensors, is_backward: bool) -> Tensor:
        """
        Read the observations of each sample.

        Parameters
        ----------
        sequences : IrregularTensors
            The padded sequences.
        is_backward : bool
            Whether the observations are read from the last to the first, ending at time 0.

        Returns
        -------
        Tensor
            The final hidden state with shape (batch, hidden_size), unchanged by padding steps.
        """
        times: Tensor = sequences.times
        inputs: Tensor = torch.cat([sequences.values, sequences.observed_mask], dim=-1)
        is_present: Tensor = sequences.observed_mask.sum(dim=-1, keepdim=True) > 0
        if is_backward:
            times, inputs, is_present = self._reverse(sequences, inputs, is_present)
        state: Tensor = times.new_zeros(times.shape[0], self.hidden_size)
        previous_time: Tensor = times[:, :1] if is_backward else torch.zeros_like(times[:, :1])
        for step in range(times.shape[1]):
            time: Tensor = times[:, step : step + 1]
            state = self.evolve(state, previous_time, time)
            state = torch.where(is_present[:, step], self.cell(inputs[:, step], state), state)
            previous_time = time
        if is_backward:
            state = self.evolve(state, previous_time, torch.zeros_like(previous_time))
        return state

    def evolve(self, state: Tensor, start_time: Tensor, end_time: Tensor) -> Tensor:
        """
        Follow the differential equation of the hidden state between two times of each sample.

        Parameters
        ----------
        state : Tensor
            Hidden states with shape (batch, hidden_size).
        start_time : Tensor
            Start times with shape (batch, 1).
        end_time : Tensor
            End times with shape (batch, 1).

        Returns
        -------
        Tensor
            The hidden states at the end times.
        """
        return self.solver.integrate(
            lambda _, hidden: self.derivative(hidden), state, start_time, end_time, self.steps_per_interval
        )

    @staticmethod
    def _reverse(sequences: IrregularTensors, inputs: Tensor, is_present: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        steps: Tensor = torch.arange(sequences.times.shape[1], device=inputs.device).unsqueeze(0)
        lengths: Tensor = sequences.lengths.unsqueeze(1).to(steps.dtype)
        is_real: Tensor = steps < lengths
        order: Tensor = torch.where(is_real, lengths - 1 - steps, steps)
        # Padding steps keep their place after the reversed real steps and take the first time, where the reversed
        # reading ends, so the solver advances by zero time over them.
        times: Tensor = torch.where(is_real, sequences.times.gather(1, order), sequences.times[:, :1])
        channel_order: Tensor = order.unsqueeze(-1)
        return (
            times,
            inputs.gather(1, channel_order.expand(-1, -1, inputs.shape[2])),
            is_present.gather(1, channel_order),
        )

    def __call__(self, sequences: IrregularTensors, is_backward: bool) -> Tensor:
        return cast(Tensor, super().__call__(sequences, is_backward))
