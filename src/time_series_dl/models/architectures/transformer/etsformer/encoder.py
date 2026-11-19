from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .encoder_layer import ETSformerEncoderLayer
from .encoding import ETSformerEncoding

if TYPE_CHECKING:
    from .parameters import ETSformerParameters


class ETSformerEncoder(nn.Module):
    """
    ETSformer encoder: a causal convolutional embedding of the values followed by layers that each extract a growth
    and a seasonality, while the level of the input channels is smoothed alongside.

    Parameters
    ----------
    parameters : ETSformerParameters
        The model configuration.
    channels : int
        Number of input channels.
    horizon : int
        Number of steps the seasonality is extrapolated; zero when only the input window is needed.
    is_level_tracked : bool
        Whether the level of the input channels is computed.

    Raises
    ------
    ValueError
        If channels is not positive or horizon is negative.
    """

    def __init__(self, parameters: ETSformerParameters, channels: int, horizon: int, is_level_tracked: bool) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if horizon < 0:
            raise ValueError(f"horizon must not be negative, got {horizon}")
        super().__init__()
        self.embedding: nn.Conv1d = nn.Conv1d(channels, parameters.model_size, kernel_size=3, padding=2, bias=False)
        _ = nn.init.kaiming_normal_(self.embedding.weight)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.is_level_tracked: bool = is_level_tracked
        self.layers: nn.ModuleList = nn.ModuleList(
            ETSformerEncoderLayer(
                parameters,
                channels,
                horizon,
                is_level_tracked=is_level_tracked,
                is_residual_returned=index < parameters.num_layers - 1,
            )
            for index in range(parameters.num_layers)
        )

    def forward(self, values: Tensor) -> ETSformerEncoding:
        """
        Encode a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        ETSformerEncoding
            The level after the last layer and the growth and seasonality of every layer.
        """
        # Padding both ends by two and dropping the last two outputs makes the convolution causal.
        embedded: Tensor = self.embedding(values.transpose(1, 2))[..., :-2]
        residual: Tensor = self.dropout(embedded.transpose(1, 2))
        level: Tensor | None = values if self.is_level_tracked else None
        growths: list[Tensor] = []
        seasons: list[Tensor] = []
        for layer in self.layers:
            output = cast(ETSformerEncoderLayer, layer)(residual, level)
            growths.append(output.growth)
            seasons.append(output.season)
            level = output.level
            if output.residual is not None:
                residual = output.residual
        return ETSformerEncoding(level=level, growths=tuple(growths), seasons=tuple(seasons))

    def __call__(self, values: Tensor) -> ETSformerEncoding:
        return cast(ETSformerEncoding, super().__call__(values))
