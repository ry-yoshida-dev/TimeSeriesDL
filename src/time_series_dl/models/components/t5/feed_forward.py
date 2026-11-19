from typing import cast

from torch import Tensor, nn

from .feed_forward_kind import T5FeedForwardKind


class T5FeedForward(nn.Module):
    """
    Position-wise feed-forward network of T5, without biases.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    feedforward_size : int
        Hidden size.
    kind : T5FeedForwardKind
        ReLU, or a tanh-approximated GELU gate times a linear projection.
    dropout : float
        Dropout probability of the hidden layer.
    """

    def __init__(self, model_size: int, feedforward_size: int, kind: T5FeedForwardKind, dropout: float) -> None:
        super().__init__()
        self.kind: T5FeedForwardKind = kind
        self.input: nn.Linear = nn.Linear(model_size, feedforward_size, bias=False)
        self.gate: nn.Linear | None = None
        match kind:
            case T5FeedForwardKind.RELU:
                self.activation: nn.Module = nn.ReLU()
            case T5FeedForwardKind.GATED_GELU:
                self.activation = nn.GELU(approximate="tanh")
                self.gate = nn.Linear(model_size, feedforward_size, bias=False)
        self.output: nn.Linear = nn.Linear(feedforward_size, model_size, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

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
        if self.gate is None:
            activated: Tensor = self.activation(self.input(hidden))
        else:
            activated = self.activation(self.gate(hidden)) * self.input(hidden)
        outputs: Tensor = self.output(self.dropout(activated))
        return outputs

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
