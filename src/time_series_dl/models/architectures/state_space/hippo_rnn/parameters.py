from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....components.state_space import Discretization, HiPPOMeasure
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .sequence_encoder import HiPPORNNSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class HiPPORNNParameters:
    """
    HiPPO-RNN (Gu et al., 2020): a gated recurrent cell with a HiPPO memory that keeps the optimal polynomial
    projection of a learned feature's history, with the generic heads of every task.

    The scaled Legendre measure remembers the whole history without a timescale; the translated Legendre measure
    gives the memory of the Legendre Memory Unit.

    Attributes
    ----------
    measure : HiPPOMeasure
        Measure over the history that the memory projects onto.
    hidden_size : int
        Size of the hidden state.
    state_size : int
        Number of basis functions of the memory.
    window_ratio : float
        Length of the window of a translated measure, relative to the input length; ignored by the scaled measure.
    discretization : Discretization
        Conversion of the continuous memory dynamics to steps.
    head : EncoderHead
        Output layers; summaries use the last hidden state by default.

    Raises
    ------
    ValueError
        If a size or window_ratio is not positive, or state_size is odd for the Fourier measure.
    """

    measure: HiPPOMeasure = HiPPOMeasure.LEGS
    hidden_size: int = 128
    state_size: int = 64
    window_ratio: float = 1.0
    discretization: Discretization = Discretization.BILINEAR
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.state_size <= 0:
            raise ValueError(f"state_size must be positive, got {self.state_size}")
        if self.window_ratio <= 0.0:
            raise ValueError(f"window_ratio must be positive, got {self.window_ratio}")
        if self.measure is HiPPOMeasure.FOUT and self.state_size % 2 != 0:
            raise ValueError(f"the Fourier measure needs an even state_size, got {self.state_size}")

    def build_encoder(self, input_length: int, input_channels: int) -> HiPPORNNSequenceEncoder:
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
        HiPPORNNSequenceEncoder
            The encoder.
        """
        return HiPPORNNSequenceEncoder(self, input_length, input_channels)

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
