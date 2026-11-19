from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn

from .de_stationary_factors import DeStationaryFactors
from .factor_projector import NonStationaryFactorProjector
from .stationarization import Stationarization


class DeStationaryFactorLearner(nn.Module):
    """
    Learns the de-stationary factors from the raw series: the scale tau as the exponential of a projection of the
    series and its standard deviation, and the shift delta of every input step from the series and its mean.

    Parameters
    ----------
    input_length : int
        Number of time steps of the raw series.
    input_channels : int
        Number of channels of the raw series.
    hidden_sizes : Sequence[int]
        Sizes of the hidden layers of both projectors.
    """

    def __init__(self, input_length: int, input_channels: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.scale_projector: NonStationaryFactorProjector = NonStationaryFactorProjector(
            input_length, input_channels, hidden_sizes, 1
        )
        self.shift_projector: NonStationaryFactorProjector = NonStationaryFactorProjector(
            input_length, input_channels, hidden_sizes, input_length
        )

    def forward(self, values: Tensor, stationarization: Stationarization) -> DeStationaryFactors:
        """
        Learn the factors of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Raw values with shape (batch, input_length, input_channels); they are detached from the graph.
        stationarization : Stationarization
            The statistics of the windows.

        Returns
        -------
        DeStationaryFactors
            The factors.
        """
        raw: Tensor = values.detach()
        scale: Tensor = self.scale_projector(raw, stationarization.standard_deviation).exp()
        shift: Tensor = self.shift_projector(raw, stationarization.mean)
        return DeStationaryFactors(scale=scale, shift=shift)

    def __call__(self, values: Tensor, stationarization: Stationarization) -> DeStationaryFactors:
        return cast(DeStationaryFactors, super().__call__(values, stationarization))
