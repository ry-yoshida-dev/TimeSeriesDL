from typing import cast

from torch import Tensor, nn


class ChronosBoltResidualBlock(nn.Module):
    """
    Residual multilayer perceptron of Chronos-Bolt that embeds input patches and projects the decoder state to the
    quantiles: a ReLU hidden layer plus a linear shortcut.

    Parameters
    ----------
    input_size : int
        Size of the inputs.
    hidden_size : int
        Size of the hidden layer.
    output_size : int
        Size of the outputs.
    dropout : float
        Dropout probability of the hidden branch.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden: nn.Linear = nn.Linear(input_size, hidden_size)
        self.output: nn.Linear = nn.Linear(hidden_size, output_size)
        self.residual: nn.Linear = nn.Linear(input_size, output_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Transform the last dimension.

        Parameters
        ----------
        values : Tensor
            Inputs with shape (..., input_size).

        Returns
        -------
        Tensor
            Outputs with shape (..., output_size).
        """
        hidden: Tensor = self.dropout(self.output(self.hidden(values).relu()))
        shortcut: Tensor = self.residual(values)
        return hidden + shortcut

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
