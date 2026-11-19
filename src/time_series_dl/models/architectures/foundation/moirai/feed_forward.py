from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class MoiraiFeedForward(nn.Module):
    """
    Gated linear unit feed-forward network of Moirai (SwiGLU, without biases).

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    feedforward_size : int
        Hidden size.
    dropout : float
        Dropout probability of the hidden layer and of the output.
    """

    def __init__(self, model_size: int, feedforward_size: int, dropout: float) -> None:
        super().__init__()
        self.input: nn.Linear = nn.Linear(model_size, feedforward_size, bias=False)
        self.gate: nn.Linear = nn.Linear(model_size, feedforward_size, bias=False)
        self.output: nn.Linear = nn.Linear(feedforward_size, model_size, bias=False)
        self.hidden_dropout: nn.Dropout = nn.Dropout(dropout)
        self.output_dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Transform every token.

        Parameters
        ----------
        hidden : Tensor
            Representations with shape (..., model_size).

        Returns
        -------
        Tensor
            The outputs with the same shape.
        """
        activated: Tensor = functional.silu(self.gate(hidden)) * self.input(hidden)
        outputs: Tensor = self.output_dropout(self.output(self.hidden_dropout(activated)))
        return outputs

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
