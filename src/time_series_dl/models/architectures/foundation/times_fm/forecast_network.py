from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .model import TimesFMModel

if TYPE_CHECKING:
    from .parameters import TimesFMParameters


class TimesFMForecastNetwork(ForecastNetwork):
    """
    TimesFM applied to each target channel as a separate univariate series; the other channels are not read.

    Horizons longer than ``output_patch_length`` are decoded autoregressively as in the official code: the mean
    forecast of each block is appended to the context, which keeps its last ``context_length`` values. With one
    parameter per value the network outputs the mean head, and with one per quantile level the quantile heads.
    Missing (NaN) values are treated as padding.

    Parameters
    ----------
    parameters : TimesFMParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If ``num_parameters`` is neither 1 nor the number of quantile levels.
    """

    def __init__(self, parameters: TimesFMParameters, shape: ForecastShape) -> None:
        levels: tuple[float, ...] = parameters.quantile_levels
        if shape.num_parameters not in (1, len(levels)):
            raise ValueError(
                f"TimesFM forecasts the mean or the {len(levels)} quantiles {levels}; "
                + f"num_parameters must be 1 or {len(levels)}, got {shape.num_parameters}"
            )
        super().__init__()
        self.model: TimesFMModel = TimesFMModel(parameters)
        self.horizon: int = shape.horizon
        self.context_length: int = parameters.context_length
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)
        self.output_indices: Tensor
        self.register_buffer(
            "output_indices",
            torch.tensor([0] if shape.num_parameters == 1 else list(range(1, len(levels) + 1))),
            persistent=False,
        )

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
        outputs: Tensor = self.forecast(series).index_select(2, self.output_indices)
        return outputs.reshape(batch, -1, self.horizon, outputs.shape[2]).transpose(1, 2)

    def forecast(self, series: Tensor) -> Tensor:
        """
        Forecast the mean and the quantiles of the whole horizon of univariate series.

        Parameters
        ----------
        series : Tensor
            Past values with shape (batch, length); NaN marks missing values.

        Returns
        -------
        Tensor
            The mean followed by the quantiles, with shape (batch, horizon, 1 + number of quantile levels).
        """
        is_padded: Tensor = torch.isnan(series)[:, -self.context_length :]
        context: Tensor = torch.nan_to_num(series[:, -self.context_length :], nan=0.0)
        blocks: list[Tensor] = []
        remaining: int = self.horizon
        while remaining > 0:
            block: Tensor = self.model(context, is_padded)
            blocks.append(block)
            remaining -= block.shape[1]
            if remaining > 0:
                context = torch.cat([context, block[..., 0]], dim=1)[:, -self.context_length :]
                is_padded = torch.cat([is_padded, torch.zeros_like(block[..., 0], dtype=torch.bool)], dim=1)[
                    :, -self.context_length :
                ]
        return torch.cat(blocks, dim=1)[:, : self.horizon]
