from dataclasses import dataclass, field

from ....interfaces import ForecastNetwork, ForecastShape
from ..xlstm import XLSTMStackParameters
from .forecast_network import XLSTMTimeForecastNetwork


@dataclass(frozen=True, kw_only=True)
class XLSTMTimeParameters:
    """
    xLSTMTime (Alharthi and Mahmood, 2024): series decomposition and linear forecasts refined by an xLSTM stack that
    reads the channels as a sequence.

    The defaults follow the official code (``xlstm.py`` of muslehal/xLSTMTime, one configuration for every
    benchmark): a moving average of 25 steps, an embedding of 256, and a stack of three sLSTM blocks with four heads,
    the sLSTM variant the paper uses for the smaller datasets such as ETTh1; setting ``stack.slstm_block_indices`` to
    an empty tuple gives the mLSTM variant used for the larger datasets.

    Attributes
    ----------
    moving_average_size : int
        Length of the moving-average window of the decomposition; odd.
    model_size : int
        Size of the channel embeddings and of the features of the stack.
    stack : XLSTMStackParameters
        The xLSTM blocks.

    Raises
    ------
    ValueError
        If a size is not positive, moving_average_size is even, or the heads or blocks of the stack do not divide the
        sizes they split.
    """

    moving_average_size: int = 25
    model_size: int = 256
    stack: XLSTMStackParameters = field(
        default_factory=lambda: XLSTMStackParameters(num_blocks=3, slstm_block_indices=(0, 1, 2))
    )

    def __post_init__(self) -> None:
        if self.moving_average_size <= 0 or self.moving_average_size % 2 == 0:
            raise ValueError(f"moving_average_size must be a positive odd number, got {self.moving_average_size}")
        if self.model_size <= 0:
            raise ValueError(f"model_size must be positive, got {self.model_size}")
        self.stack.check_size(self.model_size)

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
        return XLSTMTimeForecastNetwork(self, shape)
