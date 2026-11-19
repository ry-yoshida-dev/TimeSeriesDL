from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ETSformerEncoding:
    """
    Level, growth and seasonality extracted by the ETSformer encoder.

    Attributes
    ----------
    level : Tensor | None
        Level of each input channel after the last layer, with shape (batch, length, channels); None when the level
        is not tracked.
    growths : tuple[Tensor, ...]
        Growth of each layer with shape (batch, length + 1, model_size).
    seasons : tuple[Tensor, ...]
        Seasonality of each layer with shape (batch, length + horizon, model_size).
    """

    level: Tensor | None
    growths: tuple[Tensor, ...]
    seasons: tuple[Tensor, ...]
