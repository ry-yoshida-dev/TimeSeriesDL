from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .model import ChronosBoltModel

if TYPE_CHECKING:
    from .parameters import ChronosBoltParameters


class ChronosBoltForecastNetwork(ForecastNetwork):
    """
    Chronos-Bolt applied to each target channel as a separate univariate series; the other channels are not read.

    Horizons longer than the model's ``prediction_length`` are forecast like the official pipeline does: every
    forecast quantile is appended to the context as a separate path, and the quantiles over all paths of the next
    block are taken. With one parameter per value the network outputs the median, and with one per quantile level the
    quantiles of ``quantile_levels``.

    Parameters
    ----------
    parameters : ChronosBoltParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If ``num_parameters`` is neither 1 nor the number of quantile levels, or a point forecast is requested
        without the 0.5 level.
    """

    def __init__(self, parameters: ChronosBoltParameters, shape: ForecastShape) -> None:
        levels: tuple[float, ...] = parameters.quantile_levels
        if shape.num_parameters not in (1, len(levels)):
            raise ValueError(
                f"Chronos-Bolt forecasts the median or the {len(levels)} quantiles {levels}; "
                + f"num_parameters must be 1 or {len(levels)}, got {shape.num_parameters}"
            )
        if shape.num_parameters == 1 and 0.5 not in levels:
            raise ValueError(f"a point forecast needs the 0.5 quantile level, got {levels}")
        super().__init__()
        self.model: ChronosBoltModel = ChronosBoltModel(parameters)
        self.horizon: int = shape.horizon
        self.context_length: int = parameters.context_length
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)
        self.levels: Tensor
        self.register_buffer("levels", torch.tensor(levels), persistent=False)
        self.output_indices: Tensor
        self.register_buffer(
            "output_indices",
            torch.tensor([levels.index(0.5)] if shape.num_parameters == 1 else list(range(len(levels)))),
            persistent=False,
        )

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
        quantiles: Tensor = self.forecast(series)
        selected: Tensor = quantiles.index_select(1, self.output_indices)
        return selected.reshape(batch, -1, selected.shape[1], self.horizon).permute(0, 3, 1, 2)

    def forecast(self, series: Tensor) -> Tensor:
        """
        Forecast the quantiles of the whole horizon of univariate series.

        Parameters
        ----------
        series : Tensor
            Past values with shape (batch, length).

        Returns
        -------
        Tensor
            Quantiles with shape (batch, num_quantiles, horizon).
        """
        prediction: Tensor = self.model(series)
        blocks: list[Tensor] = [prediction]
        remaining: int = self.horizon - prediction.shape[-1]
        num_quantiles: int = prediction.shape[1]
        paths: Tensor = series[:, -self.context_length :].unsqueeze(1).expand(-1, num_quantiles, -1)
        while remaining > 0:
            paths = torch.cat([paths, prediction], dim=-1)[..., -self.context_length :]
            batch, _, length = paths.shape
            samples: Tensor = self.model(paths.reshape(batch * num_quantiles, length)).reshape(
                batch, num_quantiles * num_quantiles, -1
            )
            prediction = torch.quantile(samples, self.levels.to(samples.dtype), dim=1).transpose(0, 1)
            blocks.append(prediction)
            remaining -= prediction.shape[-1]
        return torch.cat(blocks, dim=-1)[..., : self.horizon]
