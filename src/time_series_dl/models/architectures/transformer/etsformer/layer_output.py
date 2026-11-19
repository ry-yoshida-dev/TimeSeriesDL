from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ETSformerLayerOutput:
    """
    Outputs of one ETSformer encoder layer.

    Attributes
    ----------
    residual : Tensor | None
        Residual tokens for the next layer with shape (batch, length, model_size); None for the last layer, which
        has no next layer.
    level : Tensor | None
        Updated level with shape (batch, length, channels); None when the level is not tracked.
    growth : Tensor
        Growth with shape (batch, length + 1, model_size).
    season : Tensor
        Seasonality, extrapolated over the horizon, with shape (batch, length + horizon, model_size).
    """

    residual: Tensor | None
    level: Tensor | None
    growth: Tensor
    season: Tensor
