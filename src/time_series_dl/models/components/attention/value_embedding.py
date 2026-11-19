from typing import cast

from torch import Tensor, nn

from ..layers.positional_encoding import SinusoidalPositionalEncoding


class ValueEmbedding(nn.Module):
    """
    Token embedding of the values of each time step, as in Informer: a circular convolution over three steps plus the
    sinusoidal position encoding, then dropout.

    No calendar covariates are available, so the decomposition models, which embed time stamps instead of positions in
    their papers, use the position encoding too.

    Parameters
    ----------
    input_channels : int
        Number of channels of the values.
    model_size : int
        Size of each token vector.
    dropout : float
        Dropout probability of the tokens.
    max_length : int
        Longest sequence that can be embedded.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(self, input_channels: int, model_size: int, dropout: float, max_length: int) -> None:
        if input_channels <= 0 or model_size <= 0:
            raise ValueError(f"input_channels and model_size must be positive, got {input_channels} and {model_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
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
