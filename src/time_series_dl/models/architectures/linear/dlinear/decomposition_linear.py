from typing import cast

from torch import Tensor, nn

from ....components.layers import SeriesDecomposition, TemporalLinear


class DecompositionLinear(nn.Module):
    """
    Sum of linear maps along time of the seasonal part and of the trend of a series.

    Parameters
    ----------
    kernel_size : int
        Length of the moving average that extracts the trend.
    input_length : int
        Number of input time steps.
    output_length : int
        Number of output time steps.
    channels : int
        Number of channels.
    is_individual : bool
        Whether each channel has its own linear maps.
    """

    def __init__(
        self, kernel_size: int, input_length: int, output_length: int, channels: int, is_individual: bool
    ) -> None:
        super().__init__()
        self.decomposition: SeriesDecomposition = SeriesDecomposition(kernel_size)
        self.seasonal: TemporalLinear = TemporalLinear(input_length, output_length, channels, is_individual)
        self.trend: TemporalLinear = TemporalLinear(input_length, output_length, channels, is_individual)

    def forward(self, values: Tensor) -> Tensor:
        """
        Map a batch of series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Values with shape (batch, output_length, channels).
        """
        seasonal: Tensor
        trend: Tensor
        seasonal, trend = self.decomposition(values)
        return self.seasonal(seasonal) + self.trend(trend)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
