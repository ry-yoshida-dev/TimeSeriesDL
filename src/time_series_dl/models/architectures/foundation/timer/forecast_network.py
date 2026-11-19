from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .model import TimerModel

if TYPE_CHECKING:
    from .parameters import TimerParameters


class TimerForecastNetwork(ForecastNetwork):
    """
    Timer applied to each target channel as a separate univariate series, generating point forecasts patch by patch
    like the official ``generate``; the other channels are not read.

    The window is cut to its last whole number of patches. As in the official generation with its key-value cache,
    the context is normalized by its own mean and (sample) standard deviation, each generated block is fed back
    normalized by the mean and standard deviation of that block alone, and each prediction is mapped back with the
    statistics of the block it follows; the forecast is therefore equivariant to shifting and positively scaling
    the series. A standard deviation of zero is replaced by one, where the original divides by zero.

    Parameters
    ----------
    parameters : TimerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If ``num_parameters`` is not 1 or the window is shorter than one patch.
    """

    def __init__(self, parameters: TimerParameters, shape: ForecastShape) -> None:
        if shape.num_parameters != 1:
            raise ValueError(f"Timer forecasts points only; num_parameters must be 1, got {shape.num_parameters}")
        if shape.input_length < parameters.patch_length:
            raise ValueError(
                f"input_length must be at least one patch ({parameters.patch_length}), got {shape.input_length}"
            )
        super().__init__()
        self.model: TimerModel = TimerModel(parameters)
        self.horizon: int = shape.horizon
        self.patch_length: int = parameters.patch_length
        self.context_length: int = shape.input_length // parameters.patch_length * parameters.patch_length
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
        forecast: Tensor = self.forecast(series)
        return forecast.reshape(batch, -1, self.horizon).transpose(1, 2).unsqueeze(-1)

    def forecast(self, series: Tensor) -> Tensor:
        """
        Forecast the whole horizon of univariate series.

        Parameters
        ----------
        series : Tensor
            Past values with shape (batch, length), at least one patch long.

        Returns
        -------
        Tensor
            The point forecasts with shape (batch, horizon).
        """
        block: Tensor = series[:, -self.context_length :]
        patches: Tensor = torch.empty(series.shape[0], 0, self.patch_length, dtype=series.dtype, device=series.device)
        predictions: list[Tensor] = []
        remaining: int = self.horizon
        while True:
            location, scale = self._compute_statistics(block)
            normalized: Tensor = (block - location) / scale
            patches = torch.cat([patches, normalized.unfold(-1, self.patch_length, self.patch_length)], dim=1)
            block = self.model(patches)[:, -1] * scale + location
            predictions.append(block[:, :remaining])
            remaining -= block.shape[-1]
            if remaining <= 0:
                return torch.cat(predictions, dim=-1)

    @staticmethod
    def _compute_statistics(block: Tensor) -> tuple[Tensor, Tensor]:
        location: Tensor = block.mean(dim=-1, keepdim=True)
        scale: Tensor = block.std(dim=-1, keepdim=True)
        return location, torch.where(scale == 0, 1.0, scale)
