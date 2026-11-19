from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import NHITSForecastNetwork
from .interpolation import NHITSInterpolation
from .pooling import NHITSPooling
from .stack import NHITSStack


@dataclass(frozen=True, kw_only=True)
class NHITSParameters:
    """
    N-HiTS (Challu et al., 2023): N-BEATS stacks with multi-rate input pooling and hierarchical interpolation.

    Like N-BEATS, the network forecasts each target channel from that channel's own history, with weights shared
    across channels, and ignores the channels that are not forecast. The defaults follow the paper's three stacks of
    one block with two layers of width 512, max pooling and linear interpolation, with the kernel sizes (2, 2, 1) and
    downsampling factors (4, 2, 1) of the reference implementation in neuralforecast.

    Attributes
    ----------
    stacks : tuple[NHITSStack, ...]
        The stacks in order, usually from the coarsest to the finest resolution.
    pooling : NHITSPooling
        Downsampling of the input of every block.
    interpolation : NHITSInterpolation
        Interpolation of the forecast coefficients of every block.

    Raises
    ------
    ValueError
        If there is no stack.
    """

    stacks: tuple[NHITSStack, ...] = (
        NHITSStack(pooling_kernel_size=2, downsampling_factor=4),
        NHITSStack(pooling_kernel_size=2, downsampling_factor=2),
        NHITSStack(pooling_kernel_size=1, downsampling_factor=1),
    )
    pooling: NHITSPooling = NHITSPooling.MAX
    interpolation: NHITSInterpolation = NHITSInterpolation.LINEAR

    def __post_init__(self) -> None:
        if not self.stacks:
            raise ValueError("stacks must not be empty.")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return NHITSForecastNetwork(self, shape)
