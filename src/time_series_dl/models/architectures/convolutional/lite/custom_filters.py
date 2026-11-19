from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

_DETECTION_LENGTHS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


class LITECustomFilters(nn.Module):
    """
    Fixed hand-crafted filters of LITE, followed by a ReLU: increasing and decreasing trend detectors of lengths 2 to
    64, and peak detectors of lengths 6 to 96.

    As in the official code, each filter has the same weights on every input channel, so it reads the sum of the
    channels, and keeps the length with the padding of Keras' ``same`` mode. All filters are stored centered in the
    longest length so they run as one convolution; their weights are buffers.

    Parameters
    ----------
    input_channels : int
        Number of input channels.

    Raises
    ------
    ValueError
        If input_channels is not positive.
    """

    def __init__(self, input_channels: int) -> None:
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        super().__init__()
        filters: list[Tensor] = [self._trend_filter(length, is_increasing=True) for length in _DETECTION_LENGTHS]
        filters.extend(self._trend_filter(length, is_increasing=False) for length in _DETECTION_LENGTHS)
        filters.extend(self._peak_filter(length) for length in _DETECTION_LENGTHS[1:])
        self.width: int = max(filter_.shape[0] for filter_ in filters)
        center: int = (self.width - 1) // 2
        weights: Tensor = torch.zeros(len(filters), input_channels, self.width)
        for index, filter_ in enumerate(filters):
            start: int = center - (filter_.shape[0] - 1) // 2
            weights[index, :, start : start + filter_.shape[0]] = filter_
        self.weights: Tensor
        self.register_buffer("weights", weights)

    @staticmethod
    def _trend_filter(length: int, is_increasing: bool) -> Tensor:
        signs: Tensor = torch.where(torch.arange(length) % 2 == (0 if is_increasing else 1), -1.0, 1.0)
        return signs

    @staticmethod
    def _peak_filter(length: int) -> Tensor:
        quarter: int = length // 4
        left: Tensor = torch.linspace(0.0, 1.0, quarter + 1)[1:] ** 2
        right: Tensor = left.flip(0)
        return torch.cat([-left, -right, 2 * left, 2 * right, -left, -right])

    @property
    def num_filters(self) -> int:
        """
        Get the number of filters.

        Returns
        -------
        int
            The number of output channels.
        """
        return self.weights.shape[0]

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the filters.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Rectified filter outputs with shape (batch, num_filters, length).
        """
        padded: Tensor = functional.pad(values, ((self.width - 1) // 2, self.width // 2))
        return functional.relu(functional.conv1d(padded, self.weights))

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
