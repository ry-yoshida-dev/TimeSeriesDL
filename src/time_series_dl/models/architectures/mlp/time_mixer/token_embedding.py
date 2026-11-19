from typing import cast

from torch import Tensor, nn


class TimeMixerTokenEmbedding(nn.Module):
    """
    Embedding of each time step by a circular convolution over three neighboring steps, followed by dropout.

    Parameters
    ----------
    input_channels : int
        Number of channels of the input.
    model_size : int
        Size of each token vector.
    dropout : float
        Dropout probability after the embedding.
    """

    def __init__(self, input_channels: int, model_size: int, dropout: float) -> None:
        super().__init__()
        self.convolution: nn.Conv1d = nn.Conv1d(
            input_channels, model_size, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        _ = nn.init.kaiming_normal_(self.convolution.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Embed a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, input_channels).

        Returns
        -------
        Tensor
            Tokens with shape (batch, length, model_size).
        """
        tokens: Tensor = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        dropped: Tensor = self.dropout(tokens)
        return dropped

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
