from typing import cast

import torch
from torch import Tensor, nn


class DiffusionTSTrendBlock(nn.Module):
    """
    Polynomial trend of Diffusion-TS: convolutions reduce the tokens to the coefficients of each output channel for
    the powers ``1..degree`` of the normalized time ``t / (length + 1)``, and the trend is their polynomial.

    The first convolution reads the time steps as its channels and the token features as its length; the second
    maps the features to the output channels.

    Parameters
    ----------
    length : int
        Number of time steps.
    model_size : int
        Size of each token vector.
    output_channels : int
        Number of channels of the trend.
    degree : int
        Highest power of the polynomial.
    """

    def __init__(self, length: int, model_size: int, output_channels: int, degree: int) -> None:
        super().__init__()
        self.coefficient_convolution: nn.Conv1d = nn.Conv1d(length, degree, kernel_size=3, padding=1)
        self.activation: nn.GELU = nn.GELU()
        self.channel_convolution: nn.Conv1d = nn.Conv1d(model_size, output_channels, kernel_size=3, padding=1)
        times: Tensor = torch.arange(1, length + 1, dtype=torch.float32) / (length + 1)
        self.powers: Tensor
        self.register_buffer(
            "powers", torch.stack([times ** float(power) for power in range(1, degree + 1)]), persistent=False
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Compute the trend of a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            The trend with shape (batch, length, output_channels).
        """
        coefficients: Tensor = self.activation(self.coefficient_convolution(tokens))
        # (batch, output_channels, degree) times (degree, length).
        per_channel: Tensor = self.channel_convolution(coefficients.transpose(1, 2))
        return (per_channel @ self.powers).transpose(1, 2)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
