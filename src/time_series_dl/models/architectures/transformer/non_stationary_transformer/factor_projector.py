from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from torch import Tensor, nn


class NonStationaryFactorProjector(nn.Module):
    """
    Projector of the Non-stationary Transformer that learns one de-stationary factor from the raw series and one of
    its statistics.

    A circular convolution over the channels with one filter per time step condenses the series into one vector, which
    is concatenated with the statistic and mapped by a ReLU multilayer perceptron whose last layer has no bias.

    Parameters
    ----------
    input_length : int
        Number of time steps of the raw series.
    input_channels : int
        Number of channels of the raw series.
    hidden_sizes : Sequence[int]
        Sizes of the hidden layers; at least one.
    output_size : int
        Size of the factor.

    Raises
    ------
    ValueError
        If a size is not positive or hidden_sizes is empty.
    """

    KERNEL_SIZE: ClassVar[int] = 3

    def __init__(self, input_length: int, input_channels: int, hidden_sizes: Sequence[int], output_size: int) -> None:
        sizes: list[int] = [input_length, input_channels, *hidden_sizes, output_size]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if any(size <= 0 for size in sizes):
            raise ValueError(f"sizes must be positive, got {sizes}")
        super().__init__()
        self.series_convolution: nn.Conv1d = nn.Conv1d(
            input_length,
            1,
            kernel_size=self.KERNEL_SIZE,
            padding=self.KERNEL_SIZE // 2,
            padding_mode="circular",
            bias=False,
        )
        layer_sizes: list[int] = [2 * input_channels, *hidden_sizes]
        layers: list[nn.Module] = []
        for input_size, output_layer_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            layers.extend([nn.Linear(input_size, output_layer_size), nn.ReLU()])
        layers.append(nn.Linear(layer_sizes[-1], output_size, bias=False))
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, values: Tensor, statistic: Tensor) -> Tensor:
        """
        Learn the factor of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Raw values with shape (batch, input_length, input_channels).
        statistic : Tensor
            Mean or standard deviation with shape (batch, 1, input_channels).

        Returns
        -------
        Tensor
            Factors with shape (batch, output_size).
        """
        condensed: Tensor = self.series_convolution(values)
        features: Tensor = torch.cat([condensed, statistic], dim=1).flatten(start_dim=1)
        factors: Tensor = self.layers(features)
        return factors

    def __call__(self, values: Tensor, statistic: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values, statistic))
