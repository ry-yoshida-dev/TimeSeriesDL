from dataclasses import dataclass

import torch
from torch import Tensor

from .butcher_tableau import ButcherTableau
from .vector_field import VectorField


@dataclass(frozen=True)
class ODESolver:
    """
    Fixed-step explicit Runge-Kutta solver of ordinary differential equations, batched over samples that each have
    their own start and end times.

    Every step adds ``step_size`` times a combination of derivatives to the state, so an interval of zero length
    leaves the state exactly unchanged; padded steps that repeat a time are therefore no-ops.

    Attributes
    ----------
    tableau : ButcherTableau
        The coefficients of the method.
    """

    tableau: ButcherTableau

    def step(self, field: VectorField, time: Tensor, state: Tensor, step_size: Tensor) -> Tensor:
        """
        Advance the state by one step.

        Parameters
        ----------
        field : VectorField
            The right-hand side of the equation.
        time : Tensor
            The current time, broadcastable against the state, such as shape (batch, 1).
        state : Tensor
            The current state.
        step_size : Tensor
            The step size of each sample, broadcastable against the state; it may be zero or negative.

        Returns
        -------
        Tensor
            The state one step later.
        """
        derivatives: list[Tensor] = []
        for node, row in zip(self.tableau.nodes, self.tableau.coefficients, strict=True):
            stage_state: Tensor = state
            increments: list[Tensor] = [
                coefficient * derivative
                for coefficient, derivative in zip(row, derivatives, strict=True)
                if coefficient != 0.0
            ]
            if increments:
                stage_state = state + step_size * torch.stack(increments).sum(dim=0)
            derivatives.append(field(time + node * step_size, stage_state))
        combination: Tensor = torch.stack(
            [
                weight * derivative
                for weight, derivative in zip(self.tableau.weights, derivatives, strict=True)
                if weight != 0.0
            ]
        ).sum(dim=0)
        return state + step_size * combination

    def integrate(
        self, field: VectorField, state: Tensor, start_time: Tensor, end_time: Tensor, num_steps: int
    ) -> Tensor:
        """
        Solve the equation from the start time to the end time with equal steps.

        Parameters
        ----------
        field : VectorField
            The right-hand side of the equation.
        state : Tensor
            The state at the start time.
        start_time : Tensor
            The start time of each sample, broadcastable against the state, such as shape (batch, 1).
        end_time : Tensor
            The end time of each sample, with the shape of ``start_time``; it may precede the start time.
        num_steps : int
            Number of equal steps.

        Returns
        -------
        Tensor
            The state at the end time.

        Raises
        ------
        ValueError
            If num_steps is not positive.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        step_size: Tensor = (end_time - start_time) / num_steps
        time: Tensor = start_time
        for _ in range(num_steps):
            state = self.step(field, time, state, step_size)
            time = time + step_size
        return state

    def integrate_through(
        self, field: VectorField, state: Tensor, start_time: Tensor, times: Tensor, steps_per_interval: int
    ) -> Tensor:
        """
        Solve the equation through a sequence of times of each sample.

        Parameters
        ----------
        field : VectorField
            The right-hand side of the equation, for states of shape (batch, size).
        state : Tensor
            The state at the start time with shape (batch, size).
        start_time : Tensor
            The start time of each sample with shape (batch, 1).
        times : Tensor
            Non-decreasing times of each sample with shape (batch, num_times).
        steps_per_interval : int
            Number of equal steps between consecutive times.

        Returns
        -------
        Tensor
            The state at every time with shape (batch, num_times, size).

        Raises
        ------
        ValueError
            If steps_per_interval is not positive.
        """
        if steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {steps_per_interval}")
        states: list[Tensor] = []
        previous_time: Tensor = start_time
        for index in range(times.shape[1]):
            time: Tensor = times[:, index : index + 1]
            state = self.integrate(field, state, previous_time, time, steps_per_interval)
            states.append(state)
            previous_time = time
        if not states:
            return state.new_zeros(state.shape[0], 0, *state.shape[1:])
        return torch.stack(states, dim=1)
