from dataclasses import dataclass

from ....components.layers import Activation
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import CrossformerForecastNetwork
from .reconstruction_network import CrossformerReconstructionNetwork
from .summary_network import CrossformerSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class CrossformerParameters:
    """
    Crossformer (Zhang & Yan, ICLR 2023): dimension-segment-wise embedding, two-stage attention across time and
    across channels, and a hierarchical encoder-decoder over segments merged at coarser scales.

    The defaults follow the official code.

    Attributes
    ----------
    segment_length : int
        Number of time steps of each segment.
    merge_size : int
        Number of adjacent segments merged into one by each encoder block after the first.
    num_routers : int
        Number of router tokens per segment that relay messages between channels (the paper's c).
    model_size : int
        Size of each segment token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of encoder blocks; the decoder has one layer more.
    feedforward_size : int
        Hidden size of the feed-forward networks of the two-stage attention layers.
    dropout : float
        Dropout probability of the attention weights, the sublayer outputs and the output heads.
    activation : Activation
        Activation of the feed-forward networks.

    Raises
    ------
    ValueError
        If a size is not positive, merge_size is less than 2, num_heads does not divide model_size, or dropout is not
        in [0, 1).
    """

    segment_length: int = 6
    merge_size: int = 2
    num_routers: int = 10
    model_size: int = 256
    num_heads: int = 4
    num_layers: int = 3
    feedforward_size: int = 512
    dropout: float = 0.2
    activation: Activation = Activation.GELU

    def __post_init__(self) -> None:
        for name, value in (
            ("segment_length", self.segment_length),
            ("num_routers", self.num_routers),
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.merge_size < 2:
            raise ValueError(f"merge_size must be at least 2, got {self.merge_size}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def count_segments(self, length: int) -> int:
        """
        Get the number of segments that cover a series, after padding it to a multiple of segment_length.

        Parameters
        ----------
        length : int
            Number of time steps.

        Returns
        -------
        int
            ``ceil(length / segment_length)``.

        Raises
        ------
        ValueError
            If length is not positive.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        return -(-length // self.segment_length)

    def count_scale_segments(self, input_length: int) -> tuple[int, ...]:
        """
        Get the number of segments of every encoder scale, from the embedding to the last block.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.

        Returns
        -------
        tuple[int, ...]
            ``num_layers + 1`` counts; the first two are the segments of the embedding, and each further one is the
            previous one merged by merge_size, rounded up.
        """
        counts: list[int] = [self.count_segments(input_length)] * 2
        for _ in range(self.num_layers - 1):
            counts.append(-(-counts[-1] // self.merge_size))
        return tuple(counts)

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
        return CrossformerForecastNetwork(self, shape)

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
        return CrossformerReconstructionNetwork(self, shape)

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
        return CrossformerSummaryNetwork(self, shape)
