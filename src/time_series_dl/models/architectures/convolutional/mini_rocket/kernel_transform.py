import itertools
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .....array_types import FloatArray
from ....components.random_convolution import KernelPooling, KernelTransform

_KERNEL_LENGTH: int = 9
_NUM_LARGE_WEIGHTS: int = 3
_GOLDEN_RATIO: float = (np.sqrt(5.0) + 1.0) / 2.0


class MiniRocketKernelTransform(KernelTransform):
    """
    The 84 kernels of MiniRocket at several dilations, with biases taken from quantiles of their outputs.

    Every kernel has length 9, weight 2 at three positions and -1 at the other six. As in the official multivariate
    MiniRocket, the dilations are spread exponentially up to ``(input_length - 1) / 8``, each kernel at each dilation
    sums the convolutions of a random subset of up to nine channels, and kernels alternate between padding (pooling
    every step) and no padding (pooling the steps away from the padding, or every step when there is none).

    The paper fits each bias to a quantile, from a low-discrepancy sequence, of the kernel's output on a random
    training sequence. Here there is no separate fit step: the first batch the transform sees fits the biases, from
    the output on a random sequence of that batch, and the biases are kept fixed afterwards. They are buffers, so a
    loaded checkpoint keeps them. Until then the biases are zero.

    Parameters
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    num_biases : int
        Target number of kernel, dilation and bias combinations (the ``num_features`` of MiniRocket); rounded down to
        a multiple of 84.
    max_dilations_per_kernel : int
        Largest number of dilations per kernel.
    poolings : tuple[KernelPooling, ...]
        Poolings of each combination, each giving one feature.

    Raises
    ------
    ValueError
        If a size is not positive, num_biases is below 84, or poolings is empty.
    """

    def __init__(
        self,
        input_length: int,
        input_channels: int,
        num_biases: int,
        max_dilations_per_kernel: int,
        poolings: tuple[KernelPooling, ...],
    ) -> None:
        if input_length <= 0 or input_channels <= 0 or max_dilations_per_kernel <= 0:
            raise ValueError(
                "input_length, input_channels and max_dilations_per_kernel must be positive, got "
                + f"{input_length}, {input_channels} and {max_dilations_per_kernel}"
            )
        large_weight_positions: list[tuple[int, ...]] = list(
            itertools.combinations(range(_KERNEL_LENGTH), _NUM_LARGE_WEIGHTS)
        )
        num_kernels: int = len(large_weight_positions)
        if num_biases < num_kernels:
            raise ValueError(f"num_biases must be at least {num_kernels}, got {num_biases}")
        if not poolings:
            raise ValueError("poolings must not be empty.")
        super().__init__()
        self.num_kernels: int = num_kernels
        self.poolings: tuple[KernelPooling, ...] = poolings
        self.dilations: list[int]
        self.biases_per_dilation: list[int]
        self.dilations, self.biases_per_dilation = self._spread_dilations(
            input_length, num_biases // num_kernels, max_dilations_per_kernel
        )
        num_dilations: int = len(self.dilations)
        kernels: Tensor = -torch.ones(num_kernels, _KERNEL_LENGTH)
        _ = kernels.scatter_(1, torch.tensor(large_weight_positions), 2.0)
        max_channels: int = min(input_channels, _KERNEL_LENGTH)
        num_channels: Tensor = torch.floor(2 ** (torch.rand(num_dilations, num_kernels) * np.log2(max_channels + 1)))
        channel_ranks: Tensor = torch.rand(num_dilations, num_kernels, input_channels).argsort(dim=-1).argsort(dim=-1)
        is_selected: Tensor = channel_ranks < num_channels.clamp(min=1).to(torch.int64)[..., None]
        weights: Tensor = kernels[None, :, None, :] * is_selected[..., None]
        steps: Tensor = torch.arange(input_length)
        is_valid: Tensor = torch.ones(num_dilations, num_kernels, input_length, dtype=torch.bool)
        for dilation_index, dilation in enumerate(self.dilations):
            padding: int = (_KERNEL_LENGTH - 1) // 2 * dilation
            if 2 * padding < input_length:
                is_unpadded: Tensor = (dilation_index + torch.arange(num_kernels)) % 2 == 1
                is_inner: Tensor = (steps >= padding) & (steps < input_length - padding)
                is_valid[dilation_index] = ~is_unpadded[:, None] | is_inner[None, :]
        total: int = num_kernels * sum(self.biases_per_dilation)
        quantiles: FloatArray = (np.arange(1, total + 1, dtype=np.float64) * _GOLDEN_RATIO) % 1.0
        self.weights: Tensor
        self.is_valid: Tensor
        self.quantiles: Tensor
        self.biases: Tensor
        self.is_fitted: Tensor
        self.register_buffer("weights", weights)
        self.register_buffer("is_valid", is_valid)
        self.register_buffer("quantiles", torch.from_numpy(quantiles).to(torch.float32), persistent=False)
        self.register_buffer("biases", torch.zeros(total))
        self.register_buffer("is_fitted", torch.tensor(False))

    @staticmethod
    def _spread_dilations(
        input_length: int, biases_per_kernel: int, max_dilations_per_kernel: int
    ) -> tuple[list[int], list[int]]:
        num_dilations: int = min(biases_per_kernel, max_dilations_per_kernel)
        multiplier: float = biases_per_kernel / num_dilations
        max_exponent: float = max(float(np.log2(max(input_length - 1, 1) / (_KERNEL_LENGTH - 1))), 0.0)
        exponents: FloatArray = np.logspace(0, max_exponent, num_dilations, base=2)
        dilations: NDArray[np.int64]
        counts: NDArray[np.intp]
        dilations, counts = np.unique(exponents.astype(np.int64), return_counts=True)
        biases_per_dilation: list[int] = [int(count * multiplier) for count in counts]
        for position in range(biases_per_kernel - sum(biases_per_dilation)):
            biases_per_dilation[position % len(biases_per_dilation)] += 1
        return [int(dilation) for dilation in dilations], biases_per_dilation

    @property
    @override
    def num_features(self) -> int:
        return self.biases.shape[0] * len(self.poolings)

    @override
    def forward(self, values: Tensor) -> Tensor:
        series: Tensor = values.transpose(1, 2)
        is_fitting: bool = not bool(self.is_fitted)
        features: list[Tensor] = []
        start: int = 0
        for dilation_index, (dilation, num_biases) in enumerate(
            zip(self.dilations, self.biases_per_dilation, strict=True)
        ):
            stop: int = start + self.num_kernels * num_biases
            responses: Tensor = self.convolve(series, self.weights[dilation_index], dilation)
            if is_fitting:
                self._fit_biases(responses.detach(), start, stop)
            biases: Tensor = self.biases[start:stop].view(self.num_kernels, num_biases)
            differences: Tensor = responses[:, :, None, :] - biases[None, :, :, None]
            pooled: Tensor = KernelPooling.pool(self.poolings, differences, self.is_valid[dilation_index])
            features.append(pooled.flatten(start_dim=1))
            start = stop
        if is_fitting:
            _ = self.is_fitted.fill_(True)
        return torch.cat(features, dim=1)

    def _fit_biases(self, responses: Tensor, start: int, stop: int) -> None:
        examples: Tensor = torch.randint(responses.shape[0], (self.num_kernels,), device=responses.device)
        outputs: Tensor = responses[examples, torch.arange(self.num_kernels, device=responses.device)]
        ordered: Tensor = outputs.sort(dim=-1).values
        levels: Tensor = self.quantiles[start:stop].view(self.num_kernels, -1) * (ordered.shape[-1] - 1)
        lower: Tensor = levels.floor().to(torch.int64)
        upper: Tensor = (lower + 1).clamp(max=ordered.shape[-1] - 1)
        fraction: Tensor = levels - lower
        quantiles: Tensor = ordered.gather(-1, lower) * (1 - fraction) + ordered.gather(-1, upper) * fraction
        _ = self.biases[start:stop].copy_(quantiles.flatten())
