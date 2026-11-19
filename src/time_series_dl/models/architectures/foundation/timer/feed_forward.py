from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class TimerFeedForward(nn.Module):
    """
    Gated feed-forward network of Timer without biases: a SiLU gate times a linear projection, projected back.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    feedforward_size : int
        Hidden size.
    """

    def __init__(self, model_size: int, feedforward_size: int) -> None:
        super().__init__()
        self.gate: nn.Linear = nn.Linear(model_size, feedforward_size, bias=False)
        self.input: nn.Linear = nn.Linear(model_size, feedforward_size, bias=False)
        self.output: nn.Linear = nn.Linear(feedforward_size, model_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Transform every position.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (..., model_size).

        Returns
        -------
        Tensor
            The outputs with the same shape.
        """
        outputs: Tensor = self.output(functional.silu(self.gate(hidden)) * self.input(hidden))
        return outputs

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
