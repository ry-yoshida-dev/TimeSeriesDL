from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .basis_expansion_block import BasisExpansionBlock


class DoublyResidualStacking(nn.Module):
    """
    Doubly residual stacking of basis expansion blocks, applied to each channel independently.

    Each block reads the part of the window that the backcasts of the previous blocks left unexplained, and the
    forecast is the sum of the forecasts of every block.

    Parameters
    ----------
    blocks : Sequence[BasisExpansionBlock]
        The blocks in order; a block may appear several times to share its weights.

    Raises
    ------
    ValueError
        If there is no block.
    """

    def __init__(self, blocks: Sequence[BasisExpansionBlock]) -> None:
        if not blocks:
            raise ValueError("blocks must not be empty.")
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList(blocks)

    def forward(self, values: Tensor) -> Tensor:
        """
        Forecast every channel from its own history.

        Parameters
        ----------
        values : Tensor
            Past values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Forecasts with shape (batch, horizon, channels, num_parameters).
        """
        batch_size: int = values.shape[0]
        channels: int = values.shape[2]
        residual: Tensor = values.transpose(1, 2).reshape(batch_size * channels, -1)
        forecasts: list[Tensor] = []
        for block in self.blocks:
            backcast: Tensor
            block_forecast: Tensor
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecasts.append(block_forecast)
        forecast: Tensor = torch.stack(forecasts).sum(dim=0)
        return forecast.reshape(batch_size, channels, forecast.shape[1], -1).permute(0, 3, 1, 2)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
