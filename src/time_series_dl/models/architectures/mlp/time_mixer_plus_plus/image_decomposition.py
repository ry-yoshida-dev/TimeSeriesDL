from typing import cast

from torch import Tensor, nn

from ....components.attention import FullAttentionMechanism, MultiHeadAttention


class TimeMixerPlusPlusImageDecomposition(nn.Module):
    """
    Time image decomposition of TimeMixer++: dual-axis attention splits a time image into a seasonal image, by
    attention among the phases within each period, and a trend image, by attention among the periods at each phase.

    Parameters
    ----------
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads; it divides model_size.
    dropout : float
        Dropout probability of the attention weights.
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.seasonal_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=dropout), model_size, num_heads
        )
        self.trend_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=dropout), model_size, num_heads
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Decompose a batch of time images.

        Parameters
        ----------
        images : Tensor
            Images with shape (batch, model_size, period, columns).

        Returns
        -------
        tuple[Tensor, Tensor]
            The seasonal and the trend images, each with the shape of ``images``.
        """
        batch: int
        size: int
        period: int
        columns: int
        batch, size, period, columns = images.shape
        by_column: Tensor = images.permute(0, 3, 2, 1).reshape(batch * columns, period, size)
        seasonal: Tensor = self.seasonal_attention(by_column, by_column, by_column)
        by_row: Tensor = images.permute(0, 2, 3, 1).reshape(batch * period, columns, size)
        trend: Tensor = self.trend_attention(by_row, by_row, by_row)
        return (
            seasonal.reshape(batch, columns, period, size).permute(0, 3, 2, 1),
            trend.reshape(batch, period, columns, size).permute(0, 3, 1, 2),
        )

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(images))
