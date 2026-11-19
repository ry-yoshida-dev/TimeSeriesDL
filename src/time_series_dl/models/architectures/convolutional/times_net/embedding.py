from typing import cast

from torch import Tensor, nn

from ....components.layers import SinusoidalPositionalEncoding


class TimesNetEmbedding(nn.Module):
    """
    Circular convolution of the values to tokens plus the sinusoidal position encoding, followed by dropout.

    Parameters
    ----------
    input_channels : int
        Number of channels of the values.
    model_size : int
        Size of each token.
    max_length : int
        Longest sequence that can be embedded.
    dropout : float
        Dropout probability of the tokens.
    """

    def __init__(self, input_channels: int, model_size: int, max_length: int, dropout: float) -> None:
        super().__init__()
        self.convolution: nn.Conv1d = nn.Conv1d(
            input_channels, model_size, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        _ = nn.init.kaiming_normal_(self.convolution.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.position: SinusoidalPositionalEncoding = SinusoidalPositionalEncoding(model_size, max_length)
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
        embedded: Tensor = self.dropout(self.position(tokens))
        return embedded

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
