from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .sequence_encoder import XLSTMSequenceEncoder
from .stack_parameters import XLSTMStackParameters


@dataclass(frozen=True, kw_only=True)
class XLSTMParameters:
    """
    xLSTM (Beck et al., 2024): a stack of residual mLSTM and sLSTM blocks over the embedded time steps, with the
    generic heads of every task.

    Attributes
    ----------
    model_size : int
        Size of the embedding and of the features of the blocks (128 in the official example configuration).
    stack : XLSTMStackParameters
        The blocks, including which of them are sLSTM blocks.
    head : EncoderHead
        Output layers; summaries use the last step by default, since the blocks are causal.

    Raises
    ------
    ValueError
        If model_size is not positive or the heads or blocks of the stack do not divide the sizes they split.
    """

    model_size: int = 128
    stack: XLSTMStackParameters = field(default_factory=XLSTMStackParameters)
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        if self.model_size <= 0:
            raise ValueError(f"model_size must be positive, got {self.model_size}")
        self.stack.check_size(self.model_size)

    def build_encoder(self, input_length: int, input_channels: int) -> XLSTMSequenceEncoder:
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
        XLSTMSequenceEncoder
            The encoder.
        """
        return XLSTMSequenceEncoder(self.stack.build(self.model_size), input_length, input_channels, self.model_size)

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
