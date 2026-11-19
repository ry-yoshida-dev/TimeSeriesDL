from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor

from ....components.random_convolution import KernelPooling, KernelTransform

if TYPE_CHECKING:
    from .parameters import RocketParameters

_NORMALIZATION_EPSILON: float = 1e-8
_POOLINGS: tuple[KernelPooling, ...] = (KernelPooling.PROPORTION_OF_POSITIVE_VALUES, KernelPooling.MAX)


class RocketKernelTransform(KernelTransform):
    """
    Random kernels of ROCKET, each pooled into its proportion of positive values (PPV) and its maximum.

    Each kernel draws, as in the paper and its multivariate version in sktime, a length from ``kernel_lengths``,
    normal weights centered per channel, a bias uniform in [-1, 1], a dilation ``floor(2**u)`` with ``u`` uniform in
    ``[0, log2((input_length - 1) / (length - 1))]``, padding with probability one half, and a random subset of
    ``floor(2**v)`` channels with ``v`` uniform in ``[0, log2(min(channels, length) + 1))``. Kernels are stored
    centered in the longest length, so kernels of one dilation run as a single convolution; a kernel without padding
    pools only the steps where it does not overlap the padding, or every step when there is none.

    Parameters
    ----------
    parameters : RocketParameters
        The transform configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive.
    """

    def __init__(self, parameters: RocketParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        super().__init__()
        self.is_normalized: bool = parameters.is_normalized
        self._num_features: int = 2 * parameters.num_kernels
        num_kernels: int = parameters.num_kernels
        self.width: int = max(parameters.kernel_lengths)
        choices: Tensor = torch.tensor(parameters.kernel_lengths)
        lengths: Tensor = choices[torch.randint(len(parameters.kernel_lengths), (num_kernels,))]
        channel_limit: Tensor = lengths.clamp(max=input_channels).to(torch.float64)
        num_channels: Tensor = torch.floor(
            2 ** (torch.rand(num_kernels, dtype=torch.float64) * torch.log2(channel_limit + 1))
        )
        channel_ranks: Tensor = torch.rand(num_kernels, input_channels).argsort(dim=1).argsort(dim=1)
        is_selected_channel: Tensor = channel_ranks < num_channels.clamp(min=1).to(torch.int64)[:, None]
        taps: Tensor = torch.arange(self.width)
        offsets: Tensor = (self.width - lengths) // 2
        is_tap: Tensor = (taps[None, :] >= offsets[:, None]) & (taps[None, :] < (offsets + lengths)[:, None])
        is_weight: Tensor = is_selected_channel[:, :, None] & is_tap[:, None, :]
        weights: Tensor = torch.randn(num_kernels, input_channels, self.width) * is_weight
        channel_means: Tensor = weights.sum(dim=-1, keepdim=True) / lengths[:, None, None]
        weights = (weights - channel_means) * is_weight
        biases: Tensor = torch.rand(num_kernels) * 2 - 1
        max_exponents: Tensor = torch.log2(torch.tensor(float(max(input_length - 1, 1))) / (lengths - 1)).clamp(min=0)
        dilations: Tensor = torch.floor(2 ** (torch.rand(num_kernels) * max_exponents)).to(torch.int64)
        is_padded: Tensor = torch.rand(num_kernels) < 0.5
        half_spans: Tensor = (lengths - 1) * dilations // 2
        starts: Tensor = torch.where(is_padded | (2 * half_spans >= input_length), 0, half_spans)
        steps: Tensor = torch.arange(input_length)
        is_valid: Tensor = (steps[None, :] >= starts[:, None]) & (steps[None, :] < input_length - starts[:, None])
        order: Tensor = dilations.argsort(stable=True)
        self.weights: Tensor
        self.biases: Tensor
        self.dilations: Tensor
        self.is_valid: Tensor
        self.register_buffer("weights", weights[order])
        self.register_buffer("biases", biases[order])
        self.register_buffer("dilations", dilations[order])
        self.register_buffer("is_valid", is_valid[order])

    @property
    @override
    def num_features(self) -> int:
        return self._num_features

    @override
    def forward(self, values: Tensor) -> Tensor:
        series: Tensor = values.transpose(1, 2)
        if self.is_normalized:
            series = (series - series.mean(dim=-1, keepdim=True)) / (
                series.std(dim=-1, keepdim=True, unbiased=False) + _NORMALIZATION_EPSILON
            )
        unique_dilations: Tensor
        counts: Tensor
        # The stubs of unique_consecutive do not depend on return_counts, so its result is untyped.
        unique_dilations, counts = cast(
            tuple[Tensor, Tensor], torch.unique_consecutive(self.dilations, return_counts=True)
        )
        features: list[Tensor] = []
        start: int = 0
        dilations: list[int] = cast(list[int], unique_dilations.tolist())
        sizes: list[int] = cast(list[int], counts.tolist())
        for dilation, count in zip(dilations, sizes, strict=True):
            stop: int = start + count
            responses: Tensor = self.convolve(series, self.weights[start:stop], dilation)
            differences: Tensor = (responses + self.biases[start:stop, None]).unsqueeze(2)
            pooled: Tensor = KernelPooling.pool(_POOLINGS, differences, self.is_valid[start:stop])
            features.append(pooled.flatten(start_dim=1))
            start = stop
        return torch.cat(features, dim=1)
