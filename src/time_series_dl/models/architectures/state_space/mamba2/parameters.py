from dataclasses import dataclass, field

from torch import nn

from ....components.heads import EncoderHead
from ....components.state_space import Mamba2StateSpaceLayer, StateSpaceLayer
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from ..deep_state_space import DeepStateSpaceSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class Mamba2Parameters:
    """
    Mamba-2 (Dao and Gu, 2024): residual blocks of selective state space models with a scalar state matrix
    per head, computed by the state space duality algorithm, with the generic heads of every task.

    Attributes
    ----------
    model_size : int
        Number of channels of the features.
    state_size : int
        Size of the state of each inner channel.
    expansion : int
        Ratio of the inner size of each block to model_size.
    head_size : int
        Number of inner channels per head; must divide expansion * model_size.
    convolution_size : int
        Kernel size of the causal depthwise convolution of each block.
    chunk_length : int
        Number of steps of each chunk of the state space duality algorithm.
    num_layers : int
        Number of residual blocks.
    dropout : float
        Dropout probability of each block's output.
    is_bidirectional : bool
        Whether each block also reads the sequence backward; not causal, but every step sees the whole sequence.
    head : EncoderHead
        Output layers; summaries average over time by default.

    Raises
    ------
    ValueError
        If a size is not positive, head_size does not divide the inner size, or dropout is not in [0, 1).
    """

    model_size: int = 128
    state_size: int = 64
    expansion: int = 2
    head_size: int = 32
    convolution_size: int = 4
    chunk_length: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    is_bidirectional: bool = False
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("state_size", self.state_size),
            ("expansion", self.expansion),
            ("head_size", self.head_size),
            ("convolution_size", self.convolution_size),
            ("chunk_length", self.chunk_length),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if (self.expansion * self.model_size) % self.head_size != 0:
            raise ValueError(
                f"head_size {self.head_size} must divide the inner size {self.expansion * self.model_size}"
            )

    def build_encoder(self, input_length: int, input_channels: int) -> DeepStateSpaceSequenceEncoder:
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
        DeepStateSpaceSequenceEncoder
            The encoder.
        """
        return DeepStateSpaceSequenceEncoder(
            input_length,
            input_channels,
            self.model_size,
            self.num_layers,
            self.dropout,
            self.is_bidirectional,
            build_layer=self.build_layer,
            build_normalization=lambda: nn.RMSNorm(self.model_size),
        )

    def build_layer(self) -> StateSpaceLayer:
        """
        Build one untrained state space layer.

        Returns
        -------
        StateSpaceLayer
            The layer.
        """
        return Mamba2StateSpaceLayer(
            self.model_size,
            self.state_size,
            self.expansion,
            self.head_size,
            self.convolution_size,
            self.chunk_length,
        )

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
