from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class TTMPerceptron(nn.Module):
    """
    Two-layer perceptron of the TinyTimeMixer mixers: an expanding GELU layer and a projection back, each followed by
    dropout.

    Parameters
    ----------
    size : int
        Size of the mixed last dimension.
    expansion_factor : int
        Ratio of the hidden size to size.
    dropout : float
        Dropout probability.
    """

    def __init__(self, size: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        self.hidden: nn.Linear = nn.Linear(size, size * expansion_factor)
        self.output: nn.Linear = nn.Linear(size * expansion_factor, size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Mix the last dimension.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., size).

        Returns
        -------
        Tensor
            The outputs with the same shape.
        """
        hidden: Tensor = self.dropout(functional.gelu(self.hidden(values)))
        outputs: Tensor = self.dropout(self.output(hidden))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
