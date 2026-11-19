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
from .hierarchical_contrastive_loss import HierarchicalContrastiveLoss
from .pretraining_network import TS2VecPretrainingNetwork
from .sequence_encoder import TS2VecSequenceEncoder


def _build_default_head() -> EncoderHead:
    return EncoderHead(pooling=TemporalPooling.MAX)


@dataclass(frozen=True, kw_only=True)
class TS2VecParameters:
    """
    TS2Vec (Yue et al., AAAI 2022): a dilated convolutional encoder pretrained by hierarchical contrasting of
    overlapping random crops with masked timestamps, with the generic heads of every task.

    Defaults follow the official code. The default head takes the maximum over time, as the instance-level
    representation of the paper.

    Attributes
    ----------
    hidden_size : int
        Number of channels of the input projection and of the hidden blocks.
    output_size : int
        Size of each timestamp's representation.
    num_blocks : int
        Number of hidden residual blocks (the depth of the paper); one more block maps to ``output_size``.
    kernel_size : int
        Length of the convolution kernels; odd.
    dropout : float
        Dropout probability of the representation.
    mask_ratio : float
        Probability that a timestamp is masked after the input projection during pretraining.
    alpha : float
        Weight of the instance-wise loss; the temporal loss has weight ``1 - alpha``.
    temporal_unit : int
        Number of the finest levels of the hierarchy without the temporal loss.
    head : EncoderHead
        Output layers of every task.

    Raises
    ------
    ValueError
        If a size is not positive, kernel_size is even, a probability is out of range, alpha is not in [0, 1], or
        temporal_unit is negative.
    """

    hidden_size: int = 64
    output_size: int = 320
    num_blocks: int = 10
    kernel_size: int = 3
    dropout: float = 0.1
    mask_ratio: float = 0.5
    alpha: float = 0.5
    temporal_unit: int = 0
    head: EncoderHead = field(default_factory=_build_default_head)

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("output_size", self.output_size),
            ("num_blocks", self.num_blocks),
            ("kernel_size", self.kernel_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to keep the length, got {self.kernel_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temporal_unit < 0:
            raise ValueError(f"temporal_unit must not be negative, got {self.temporal_unit}")

    def build_encoder(self, input_length: int, input_channels: int) -> TS2VecSequenceEncoder:
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
        TS2VecSequenceEncoder
            The encoder.
        """
        return TS2VecSequenceEncoder(
            input_length,
            input_channels,
            self.hidden_size,
            self.output_size,
            self.num_blocks,
            self.kernel_size,
            self.dropout,
        )

    def build_pretraining_network(self, shape: SequenceShape) -> TS2VecPretrainingNetwork:
        """
        Build an untrained pretraining network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the unlabeled sequences.

        Returns
        -------
        TS2VecPretrainingNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences are shorter than the shortest crop.
        """
        return TS2VecPretrainingNetwork(
            self.build_encoder(shape.length, shape.channels),
            self.mask_ratio,
            HierarchicalContrastiveLoss(alpha=self.alpha, temporal_unit=self.temporal_unit),
            shape.length,
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
