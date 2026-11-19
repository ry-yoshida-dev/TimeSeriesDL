from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import TimeMixerPlusPlusForecastNetwork
from .reconstruction_network import TimeMixerPlusPlusReconstructionNetwork
from .summary_network import TimeMixerPlusPlusSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class TimeMixerPlusPlusParameters:
    """
    TimeMixer++ (Wang et al., 2025): multi-scale series turned into time images at the dominant periods, decomposed
    into seasonal and trend images by dual-axis attention, mixed across scales and across periods.

    The number of down-samplings, blocks and periods follow the long-term forecasting setting of the paper; the sizes
    are chosen in the range the paper searches.

    Attributes
    ----------
    model_size : int
        Size of each token.
    num_heads : int
        Number of heads of the dual-axis attention; it divides model_size.
    num_layers : int
        Number of mixer blocks.
    num_down_samplings : int
        Number of strided convolutions, each of which adds a coarser scale (the paper's M).
    top_k : int
        Largest number of periods of the time images.
    dropout : float
        Dropout probability of the embedding, of the attention weights and of the summary head.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    model_size: int = 32
    num_heads: int = 4
    num_layers: int = 2
    num_down_samplings: int = 3
    top_k: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("num_down_samplings", self.num_down_samplings),
            ("top_k", self.top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads must divide model_size, got {self.num_heads} and {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def scale_lengths(self, length: int) -> tuple[int, ...]:
        """
        Get the length of every scale, from the input to the coarsest.

        Each strided convolution halves the length, rounding up.

        Parameters
        ----------
        length : int
            Number of time steps of the input.

        Returns
        -------
        tuple[int, ...]
            The ``num_down_samplings + 1`` lengths.

        Raises
        ------
        ValueError
            If the coarsest scale is shorter than two steps, so it has no period to detect.
        """
        lengths: list[int] = [length]
        for _ in range(self.num_down_samplings):
            lengths.append((lengths[-1] + 1) // 2)
        if lengths[-1] < 2:
            raise ValueError(
                f"input length {length} is too short for {self.num_down_samplings} down-samplings: "
                + f"the coarsest scale has {lengths[-1]} step(s), and at least 2 are needed"
            )
        return tuple(lengths)

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
            If the input is too short for the down-samplings.
        """
        return TimeMixerPlusPlusForecastNetwork(self, shape)

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

        Raises
        ------
        ValueError
            If the sequences are too short for the down-samplings.
        """
        return TimeMixerPlusPlusReconstructionNetwork(self, shape)

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

        Raises
        ------
        ValueError
            If the input is too short for the down-samplings.
        """
        return TimeMixerPlusPlusSummaryNetwork(self, shape)
