from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....interfaces import ForecastNetwork, ForecastShape, SequenceShape, SummaryNetwork, SummaryShape
from .augmentation import TSTCCAugmentation
from .contextual_contrastive_loss import ContextualContrastiveLoss
from .pretraining_network import TSTCCPretrainingNetwork
from .sequence_encoder import TSTCCSequenceEncoder
from .temporal_contrasting import TSTCCTemporalContrasting


def _build_default_head() -> EncoderHead:
    return EncoderHead(pooling=TemporalPooling.FLATTEN)


@dataclass(frozen=True, kw_only=True)
class TSTCCParameters:
    """
    TS-TCC (Eldele et al., IJCAI 2021): a three-block convolutional encoder pretrained by temporal and contextual
    contrasting of weak and strong augmentations, with the generic forecast and summary heads.

    Defaults follow the official configuration of the HAR dataset. The default head flattens the features, as the
    official linear evaluation. The encoder shortens the sequence, so there is no reconstruction network.

    Attributes
    ----------
    channel_sizes : tuple[int, ...]
        Number of channels of each convolutional block; the last is the feature size.
    kernel_size : int
        Length of the convolution kernels.
    stride : int
        Stride of the first convolution.
    dropout : float
        Dropout probability after the first block.
    prediction_steps : int
        Number of future steps that temporal contrasting predicts.
    context_size : int
        Size of the tokens and the context vector of the temporal contrasting Transformer.
    context_layers : int
        Number of layers of the Transformer.
    context_heads : int
        Number of attention heads of the Transformer; must divide context_size.
    context_feedforward_size : int
        Hidden size of the Transformer's feed-forward networks.
    context_dropout : float
        Dropout probability of the Transformer.
    temperature : float
        Temperature of contextual contrasting.
    temporal_weight : float
        Weight of the temporal contrasting losses (lambda 1 of the paper).
    contextual_weight : float
        Weight of the contextual contrasting loss (lambda 2 of the paper).
    augmentation : TSTCCAugmentation
        The weak and strong augmentations.
    head : EncoderHead
        Output layers of forecasting, classification and regression.

    Raises
    ------
    ValueError
        If a size is not positive, the feature size is below 4, context_heads does not divide context_size, a
        dropout is not in [0, 1), or a weight is negative.
    """

    channel_sizes: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 8
    stride: int = 1
    dropout: float = 0.35
    prediction_steps: int = 6
    context_size: int = 100
    context_layers: int = 4
    context_heads: int = 4
    context_feedforward_size: int = 64
    context_dropout: float = 0.1
    temperature: float = 0.2
    temporal_weight: float = 1.0
    contextual_weight: float = 0.7
    augmentation: TSTCCAugmentation = field(default_factory=TSTCCAugmentation)
    head: EncoderHead = field(default_factory=_build_default_head)

    def __post_init__(self) -> None:
        if not self.channel_sizes or any(size <= 0 for size in self.channel_sizes):
            raise ValueError(f"channel_sizes must be non-empty and positive, got {self.channel_sizes}")
        if self.channel_sizes[-1] < 4:
            raise ValueError(f"the last channel size must be at least 4, got {self.channel_sizes[-1]}")
        for name, value in (
            ("kernel_size", self.kernel_size),
            ("stride", self.stride),
            ("prediction_steps", self.prediction_steps),
            ("context_size", self.context_size),
            ("context_layers", self.context_layers),
            ("context_heads", self.context_heads),
            ("context_feedforward_size", self.context_feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.context_size % self.context_heads != 0:
            raise ValueError(f"context_heads {self.context_heads} must divide context_size {self.context_size}")
        for name, probability in (("dropout", self.dropout), ("context_dropout", self.context_dropout)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {probability}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temporal_weight < 0.0 or self.contextual_weight < 0.0:
            raise ValueError(f"weights must not be negative, got {self.temporal_weight} and {self.contextual_weight}")

    def build_encoder(self, input_length: int, input_channels: int) -> TSTCCSequenceEncoder:
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
        TSTCCSequenceEncoder
            The encoder.
        """
        return TSTCCSequenceEncoder(
            input_length, input_channels, self.channel_sizes, self.kernel_size, self.stride, self.dropout
        )

    def build_pretraining_network(self, shape: SequenceShape) -> TSTCCPretrainingNetwork:
        """
        Build an untrained pretraining network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the unlabeled sequences.

        Returns
        -------
        TSTCCPretrainingNetwork
            The network.

        Raises
        ------
        ValueError
            If the encoded sequences are not longer than prediction_steps.
        """
        return TSTCCPretrainingNetwork(
            self.build_encoder(shape.length, shape.channels),
            TSTCCTemporalContrasting(
                self.channel_sizes[-1],
                self.context_size,
                self.context_layers,
                self.context_heads,
                self.context_feedforward_size,
                self.context_dropout,
                self.prediction_steps,
            ),
            self.augmentation,
            ContextualContrastiveLoss(temperature=self.temperature),
            self.context_size,
            self.temporal_weight,
            self.contextual_weight,
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
