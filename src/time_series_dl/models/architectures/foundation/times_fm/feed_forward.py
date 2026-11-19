from typing import cast

from torch import Tensor, nn


class TimesFMFeedForward(nn.Module):
    """
    Feed-forward network of a TimesFM layer: layer normalization, a ReLU hidden layer and a residual connection, with
    the outputs of padded patches zeroed before the residual is added.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    feedforward_size : int
        Hidden size.
    """

    def __init__(self, model_size: int, feedforward_size: int) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(model_size, eps=1e-6)
        self.hidden: nn.Linear = nn.Linear(model_size, feedforward_size)
        self.output: nn.Linear = nn.Linear(feedforward_size, model_size)

    def forward(self, hidden: Tensor, is_padded: Tensor) -> Tensor:
        """
        Transform every patch.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).
        is_padded : Tensor
            1.0 at fully padded patches and 0.0 elsewhere, with shape (batch, length).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        transformed: Tensor = self.output(self.hidden(self.norm(hidden)).relu())
        return transformed * (1.0 - is_padded.unsqueeze(-1)) + hidden

    def __call__(self, hidden: Tensor, is_padded: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, is_padded))
