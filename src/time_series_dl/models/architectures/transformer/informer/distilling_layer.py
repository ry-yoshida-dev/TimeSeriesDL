from typing import cast

from torch import Tensor, nn


class InformerDistillingLayer(nn.Module):
    """
    Self-attention distilling of Informer: a circular convolution over three steps, batch normalization, ELU and a
    max-pooling with stride 2 that halves the length.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    """

    def __init__(self, model_size: int) -> None:
        super().__init__()
        self.convolution: nn.Conv1d = nn.Conv1d(
            model_size, model_size, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(model_size)
        self.activation: nn.ELU = nn.ELU()
        self.pooling: nn.MaxPool1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def output_length(length: int) -> int:
        """
        Get the length of a distilled sequence.

        Parameters
        ----------
        length : int
            Number of input time steps.

        Returns
        -------
        int
            ``ceil(length / 2)``.
        """
        return (length + 1) // 2

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Distill a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Tokens with shape (batch, output_length(length), model_size).
        """
        hidden: Tensor = self.activation(self.norm(self.convolution(tokens.transpose(1, 2))))
        distilled: Tensor = self.pooling(hidden)
        return distilled.transpose(1, 2)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
