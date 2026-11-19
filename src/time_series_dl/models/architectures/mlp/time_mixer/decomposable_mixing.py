from typing import cast

from torch import Tensor, nn

from ....components.layers import Activation, MultilayerPerceptron, SeriesDecomposition
from .season_mixing import TimeMixerSeasonMixing
from .trend_mixing import TimeMixerTrendMixing


class TimeMixerDecomposableMixing(nn.Module):
    """
    Past decomposable mixing block: the tokens of every scale are decomposed into seasonal parts, mixed bottom-up,
    and trends, mixed top-down, and their sum passes a residual feed-forward network.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of time steps of each scale, from the finest to the coarsest.
    model_size : int
        Size of each token vector.
    feedforward_size : int
        Size of the hidden layer of the feed-forward networks.
    kernel_size : int
        Length of the moving average that extracts the trends.
    is_channel_mixing : bool
        Whether the tokens embed several channels, so a feed-forward network also mixes the seasonal parts and the
        trends before the scales are mixed.
    is_finest_output_only : bool
        Whether only the finest scale is returned, which needs no bottom-up mixing of the seasonal parts.
    """

    def __init__(
        self,
        lengths: tuple[int, ...],
        model_size: int,
        feedforward_size: int,
        kernel_size: int,
        is_channel_mixing: bool,
        is_finest_output_only: bool,
    ) -> None:
        super().__init__()
        self.decomposition: SeriesDecomposition = SeriesDecomposition(kernel_size)
        self.cross: MultilayerPerceptron | None = (
            MultilayerPerceptron(model_size, [feedforward_size], model_size, Activation.GELU)
            if is_channel_mixing
            else None
        )
        self.season_mixing: TimeMixerSeasonMixing | None = (
            None if is_finest_output_only else TimeMixerSeasonMixing(lengths)
        )
        self.trend_mixing: TimeMixerTrendMixing = TimeMixerTrendMixing(lengths)
        self.feedforward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, [feedforward_size], model_size, Activation.GELU
        )

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Mix the tokens of every scale.

        Parameters
        ----------
        tokens : list[Tensor]
            Tokens with shape (batch, length of the scale, model_size), from the finest scale.

        Returns
        -------
        list[Tensor]
            The mixed tokens with the same shapes, or only those of the finest scale.
        """
        seasons: list[Tensor] = []
        trends: list[Tensor] = []
        cross: MultilayerPerceptron | None = self.cross
        for scale_tokens in tokens:
            season: Tensor
            trend: Tensor
            season, trend = self.decomposition(scale_tokens)
            if cross is not None:
                season = cross(season)
                trend = cross(trend)
            seasons.append(season.transpose(1, 2))
            trends.append(trend.transpose(1, 2))
        mixed_seasons: list[Tensor] = seasons[:1] if self.season_mixing is None else self.season_mixing(seasons)
        mixed_trends: list[Tensor] = self.trend_mixing(trends)
        return [
            scale_tokens + self.feedforward((season + trend).transpose(1, 2))
            for scale_tokens, season, trend in zip(tokens, mixed_seasons, mixed_trends, strict=False)
        ]

    def __call__(self, tokens: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(tokens))
