from typing import cast

from torch import Tensor, nn


class ConvTranConvolutionalEmbedding(nn.Module):
    """
    Convolutional token embedding of ConvTran: a temporal convolution of each channel into ``4 * model_size``
    features, then a convolution across all channels into ``model_size`` features, each followed by batch
    normalization and GELU.

    Parameters
    ----------
    input_channels : int
        Number of channels of the values.
    model_size : int
        Size of each token vector.
    kernel_size : int
        Number of time steps of the temporal convolution, which keeps the length by zero padding, with the extra step
        of an even kernel on the right as PyTorch's ``same`` padding.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_channels: int, model_size: int, kernel_size: int) -> None:
        if input_channels <= 0 or model_size <= 0 or kernel_size <= 0:
            raise ValueError(
                "input_channels, model_size and kernel_size must be positive, got "
                + f"{input_channels}, {model_size} and {kernel_size}"
            )
        super().__init__()
        hidden_size: int = 4 * model_size
        self.temporal: nn.Sequential = nn.Sequential(
            nn.ZeroPad2d(((kernel_size - 1) // 2, kernel_size // 2, 0, 0)),
            nn.Conv2d(1, hidden_size, kernel_size=(1, kernel_size)),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
        )
        self.spatial: nn.Sequential = nn.Sequential(
            nn.Conv2d(hidden_size, model_size, kernel_size=(input_channels, 1)),
            nn.BatchNorm2d(model_size),
            nn.GELU(),
        )

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
        features: Tensor = self.temporal(values.transpose(1, 2).unsqueeze(1))
        tokens: Tensor = self.spatial(features).squeeze(2).transpose(1, 2)
        return tokens

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
