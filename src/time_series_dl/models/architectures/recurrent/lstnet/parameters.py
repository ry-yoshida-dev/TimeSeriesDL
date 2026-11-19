from dataclasses import dataclass

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....interfaces import ForecastNetwork, ForecastShape, SummaryNetwork, SummaryShape
from .forecast_network import LSTNetForecastNetwork
from .sequence_encoder import LSTNetSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class LSTNetParameters:
    """
    LSTNet (Lai et al., 2018): convolution, GRU and recurrent-skip GRU over the window, with a highway autoregressive
    component for forecasts.

    The defaults are those of the official code (100 convolution channels with kernel 6, a GRU of 100, a skip GRU of
    5 with period 24, a highway window of 24, dropout 0.2).

    Attributes
    ----------
    convolution_channels : int
        Number of output channels of the convolution.
    convolution_size : int
        Kernel size of the convolution; at most the input length.
    recurrent_size : int
        Size of the GRU's hidden state.
    skip_recurrent_size : int
        Size of the recurrent-skip GRU's hidden state.
    skip_period : int
        Period of the recurrent-skip component, which reads steps this far apart; 0 disables it. The convolved window
        must hold at least one whole period.
    highway_window : int
        Number of last values the autoregressive component reads; 0 disables it.
    dropout : float
        Dropout probability after the convolution and the recurrent components.

    Raises
    ------
    ValueError
        If a size is not positive, the period or window is negative, or dropout is not in [0, 1).
    """

    convolution_channels: int = 100
    convolution_size: int = 6
    recurrent_size: int = 100
    skip_recurrent_size: int = 5
    skip_period: int = 24
    highway_window: int = 24
    dropout: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("convolution_channels", self.convolution_channels),
            ("convolution_size", self.convolution_size),
            ("recurrent_size", self.recurrent_size),
            ("skip_recurrent_size", self.skip_recurrent_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.skip_period < 0:
            raise ValueError(f"skip_period must not be negative, got {self.skip_period}")
        if self.highway_window < 0:
            raise ValueError(f"highway_window must not be negative, got {self.highway_window}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

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

        Raises
        ------
        ValueError
            If the kernel, the skip period or the highway window does not fit in the input window.
        """
        return LSTNetForecastNetwork(self, shape)

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build an untrained summary network: the encoded features mapped linearly to the output vector.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.

        Raises
        ------
        ValueError
            If the kernel or the skip period does not fit in the input window.
        """
        encoder: LSTNetSequenceEncoder = LSTNetSequenceEncoder(self, shape.input_length, shape.input_channels)
        return EncoderHead(pooling=TemporalPooling.LAST).build_summary_network(encoder, shape)
