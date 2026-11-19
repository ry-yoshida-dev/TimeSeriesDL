from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .frequency_attention import ETSformerFrequencyAttention
from .growth_layer import ETSformerGrowthLayer
from .layer_output import ETSformerLayerOutput
from .level_layer import ETSformerLevelLayer

if TYPE_CHECKING:
    from .parameters import ETSformerParameters


class ETSformerEncoderLayer(nn.Module):
    """
    ETSformer encoder layer: frequency attention extracts the seasonality, exponential smoothing attention the growth
    of the deseasonalized residual, and the residual without both passes through a sigmoid feed-forward network, with
    layer normalization, to the next layer. The level of the input channels is updated from the growth and
    seasonality.

    Parameters
    ----------
    parameters : ETSformerParameters
        The model configuration.
    channels : int
        Number of input channels, whose level is tracked.
    horizon : int
        Number of steps the seasonality is extrapolated.
    is_level_tracked : bool
        Whether the layer updates the level.
    is_residual_returned : bool
        Whether the layer computes the residual of the next layer; False for the last layer.
    """

    def __init__(
        self,
        parameters: ETSformerParameters,
        channels: int,
        horizon: int,
        is_level_tracked: bool,
        is_residual_returned: bool,
    ) -> None:
        super().__init__()
        self.season_layer: ETSformerFrequencyAttention = ETSformerFrequencyAttention(parameters.top_k, horizon)
        self.season_dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.growth_layer: ETSformerGrowthLayer = ETSformerGrowthLayer(
            parameters.model_size, parameters.num_heads, parameters.dropout
        )
        self.growth_dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.level_layer: ETSformerLevelLayer | None = (
            ETSformerLevelLayer(parameters.model_size, channels, parameters.dropout) if is_level_tracked else None
        )
        self.feed_forward: nn.Sequential | None = None
        self.growth_norm: nn.LayerNorm | None = None
        self.feed_forward_norm: nn.LayerNorm | None = None
        if is_residual_returned:
            self.feed_forward = nn.Sequential(
                nn.Linear(parameters.model_size, parameters.feedforward_size, bias=False),
                nn.Sigmoid(),
                nn.Dropout(parameters.dropout),
                nn.Linear(parameters.feedforward_size, parameters.model_size, bias=False),
                nn.Dropout(parameters.dropout),
            )
            self.growth_norm = nn.LayerNorm(parameters.model_size)
            self.feed_forward_norm = nn.LayerNorm(parameters.model_size)

    def forward(self, residual: Tensor, level: Tensor | None) -> ETSformerLayerOutput:
        """
        Apply the layer.

        Parameters
        ----------
        residual : Tensor
            Residual tokens with shape (batch, length, model_size).
        level : Tensor | None
            Level with shape (batch, length, channels), or None when it is not tracked.

        Returns
        -------
        ETSformerLayerOutput
            The next residual, the updated level, and the growth and seasonality of the layer.
        """
        length: int = residual.shape[1]
        season: Tensor = self.season_dropout(self.season_layer(residual))
        deseasonalized: Tensor = residual - season[:, :length]
        growth: Tensor = self.growth_dropout(self.growth_layer(deseasonalized))
        next_residual: Tensor | None = None
        if self.feed_forward is not None and self.growth_norm is not None and self.feed_forward_norm is not None:
            detrended: Tensor = self.growth_norm(deseasonalized - growth[:, 1:])
            next_residual = self.feed_forward_norm(detrended + self.feed_forward(detrended))
        next_level: Tensor | None = None
        if self.level_layer is not None and level is not None:
            next_level = self.level_layer(level, growth[:, :length], season[:, :length])
        return ETSformerLayerOutput(residual=next_residual, level=next_level, growth=growth, season=season)

    def __call__(self, residual: Tensor, level: Tensor | None) -> ETSformerLayerOutput:
        return cast(ETSformerLayerOutput, super().__call__(residual, level))
