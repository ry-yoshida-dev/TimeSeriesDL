from typing import cast

import torch
from torch import Tensor, nn


class KoopaDynamicModeDecomposition(nn.Module):
    """
    One-step dynamic mode decomposition of Koopa: a linear operator ``K`` fitted by least squares so that each state of
    a sequence times ``K`` gives the next one, which reconstructs the sequence and rolls it forward.
    """

    def forward(self, states: Tensor, num_steps: int) -> tuple[Tensor, Tensor]:
        """
        Fit the operator of each sequence, reconstruct the sequence and predict the next states.

        Parameters
        ----------
        states : Tensor
            States with shape (batch, length, size); length is at least 2.
        num_steps : int
            Number of states to predict.

        Returns
        -------
        tuple[Tensor, Tensor]
            The reconstruction with shape (batch, length, size), whose first state is the input's, and the predicted
            states with shape (batch, num_steps, size).

        Raises
        ------
        ValueError
            If the sequence has fewer than 2 states or num_steps is not positive.
        """
        if states.shape[1] < 2:
            raise ValueError(f"the sequence needs at least 2 states, got {states.shape[1]}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        current: Tensor = states[:, :-1]
        operator: Tensor = cast(Tensor, torch.linalg.pinv(current)) @ states[:, 1:]
        reconstruction: Tensor = torch.cat([states[:, :1], current @ operator], dim=1)
        predictions: list[Tensor] = [states[:, -1:] @ operator]
        # Each predicted state is the previous one times the operator, an inherently recurrent rollout.
        for _ in range(num_steps - 1):
            predictions.append(predictions[-1] @ operator)
        return reconstruction, torch.cat(predictions, dim=1)

    def __call__(self, states: Tensor, num_steps: int) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(states, num_steps))
