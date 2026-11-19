import math
from dataclasses import dataclass, field

from torch import nn

from ....components.heads import EncoderHead
from ....components.state_space import LRUStateSpaceLayer, StateSpaceLayer
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
class LRUParameters:
    """
    Deep Linear Recurrent Unit (Orvieto et al., 2023): residual blocks of complex diagonal linear recurrences
    parametrized in discrete time, with the generic heads of every task.

    Attributes
    ----------
    model_size : int
        Number of channels of the features.
    state_size : int
        Size of the complex state of each layer.
    min_radius : float
        Smallest initial modulus of the eigenvalues.
    max_radius : float
        Largest initial modulus of the eigenvalues; moduli are sampled uniformly on the ring between the two.
    max_phase : float
        Largest initial phase of the eigenvalues.
    num_layers : int
        Number of residual blocks.
    dropout : float
        Dropout probability of the layers' output stage and of each block's output.
    is_bidirectional : bool
        Whether each block also reads the sequence backward; not causal, but every step sees the whole sequence.
    head : EncoderHead
        Output layers; summaries average over time by default.

    Raises
    ------
    ValueError
        If a size is not positive, the radii are not 0 < min_radius <= max_radius < 1, max_phase is not
        positive, or dropout is not in [0, 1).
    """

    model_size: int = 128
    state_size: int = 64
    min_radius: float = 0.9
    max_radius: float = 0.999
    max_phase: float = 2.0 * math.pi
    num_layers: int = 4
    dropout: float = 0.1
    is_bidirectional: bool = False
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("state_size", self.state_size),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 < self.min_radius <= self.max_radius < 1.0:
            raise ValueError(
                f"radii must satisfy 0 < min_radius <= max_radius < 1, got {self.min_radius} and {self.max_radius}"
            )
        if self.max_phase <= 0.0:
            raise ValueError(f"max_phase must be positive, got {self.max_phase}")

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
            build_normalization=lambda: nn.LayerNorm(self.model_size),
        )

    def build_layer(self) -> StateSpaceLayer:
        """
        Build one untrained state space layer.

        Returns
        -------
        StateSpaceLayer
            The layer.
        """
        return LRUStateSpaceLayer(
            self.model_size, self.state_size, self.min_radius, self.max_radius, self.max_phase, self.dropout
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
