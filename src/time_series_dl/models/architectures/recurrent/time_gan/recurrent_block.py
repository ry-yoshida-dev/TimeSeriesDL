from typing import cast

import torch
from torch import Tensor, nn

from ..vanilla import RecurrentCell


class TimeGANRecurrentBlock(nn.Module):
    """
    Stacked recurrent layers followed by a linear map of every step, optionally squashed by a sigmoid: the shape of
    each of the five TimeGAN networks.

    Parameters
    ----------
    cell : RecurrentCell
        Recurrent cell.
    input_size : int
        Size of the input vector of each step.
    hidden_size : int
        Size of the hidden state.
    num_layers : int
        Number of stacked recurrent layers.
    output_size : int
        Size of the output vector of each step.
    is_bounded : bool
        Whether the outputs pass through a sigmoid, so they lie in (0, 1).
    """

    def __init__(
        self,
        cell: RecurrentCell,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        is_bounded: bool,
    ) -> None:
        super().__init__()
        self.recurrent: nn.LSTM | nn.GRU | nn.RNN = cell.build(input_size, hidden_size, num_layers, 0.0, False)
        self.output: nn.Linear = nn.Linear(hidden_size, output_size)
        self.is_bounded: bool = is_bounded

    def forward(self, values: Tensor) -> Tensor:
        """
        Map every step of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Sequences with shape (batch, length, input_size).

        Returns
        -------
        Tensor
            Outputs with shape (batch, length, output_size).
        """
        outputs: Tensor = self.output(self.recurrent(values)[0])
        return torch.sigmoid(outputs) if self.is_bounded else outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
