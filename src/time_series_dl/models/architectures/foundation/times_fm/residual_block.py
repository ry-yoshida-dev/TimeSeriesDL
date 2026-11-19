from typing import cast

from torch import Tensor, nn


class TimesFMResidualBlock(nn.Module):
    """
    Residual multilayer perceptron of TimesFM that embeds input patches and projects the transformer outputs to the
    forecasts: a SiLU hidden layer plus a linear shortcut.

    Parameters
    ----------
    input_size : int
        Size of the inputs.
    hidden_size : int
        Size of the hidden layer.
    output_size : int
        Size of the outputs.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden: nn.Linear = nn.Linear(input_size, hidden_size)
        self.output: nn.Linear = nn.Linear(hidden_size, output_size)
        self.residual: nn.Linear = nn.Linear(input_size, output_size)

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
        hidden: Tensor = self.output(nn.functional.silu(self.hidden(values)))
        shortcut: Tensor = self.residual(values)
        return hidden + shortcut

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
