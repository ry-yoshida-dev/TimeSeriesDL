from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....components.state_space import Discretization
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .sequence_encoder import LMUSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class LMUParameters:
    """
    Legendre Memory Unit (Voelker et al., 2019): a recurrent cell whose memory projects a learned encoding of the
    last window of steps onto Legendre polynomials, with the generic heads of every task.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden state.
    state_size : int
        Order of the memory, the number of Legendre polynomials.
    window_ratio : float
        Length of the memory window, relative to the input length.
    discretization : Discretization
        Conversion of the continuous memory dynamics to steps.
    head : EncoderHead
        Output layers; summaries use the last hidden state by default.

    Raises
    ------
    ValueError
        If a size or window_ratio is not positive.
    """

    hidden_size: int = 128
    state_size: int = 64
    window_ratio: float = 1.0
    discretization: Discretization = Discretization.ZERO_ORDER_HOLD
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.state_size <= 0:
            raise ValueError(f"state_size must be positive, got {self.state_size}")
        if self.window_ratio <= 0.0:
            raise ValueError(f"window_ratio must be positive, got {self.window_ratio}")

    def build_encoder(self, input_length: int, input_channels: int) -> LMUSequenceEncoder:
        """
        Build the untrained encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        LMUSequenceEncoder
            The encoder.
        """
        return LMUSequenceEncoder(self, input_length, input_channels)

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
        return self.head.build_forecast_network(self.build_encoder(shape.input_length, shape.input_channels), shape)

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build an untrained reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.
        """
        return self.head.build_reconstruction_network(self.build_encoder(shape.length, shape.channels), shape)

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build an untrained summary network.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.
        """
        return self.head.build_summary_network(self.build_encoder(shape.input_length, shape.input_channels), shape)
