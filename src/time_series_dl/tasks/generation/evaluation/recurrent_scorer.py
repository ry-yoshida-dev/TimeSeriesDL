from typing import cast

from torch import Tensor, nn


class RecurrentScorer(nn.Module):
    """
    Small post-hoc GRU with a linear output at every step, trained from scratch to judge synthetic sequences: as the
    real-versus-synthetic classifier of the discriminative score (read at the last step) and as the next-step
    predictor of the predictive score.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    hidden_size : int
        Size of the hidden state.
    output_size : int
        Size of the output of each step.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.recurrent: nn.GRU = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Score every step of a batch of sequences.

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
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
