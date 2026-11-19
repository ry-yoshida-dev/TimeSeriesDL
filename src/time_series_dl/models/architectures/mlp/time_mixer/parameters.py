from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import TimeMixerForecastNetwork
from .reconstruction_network import TimeMixerReconstructionNetwork
from .summary_network import TimeMixerSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class TimeMixerParameters:
    """
    TimeMixer (Wang et al., 2024): the window is average-pooled into several scales, past decomposable mixing blocks
    mix the seasonal parts bottom-up and the trends top-down across scales, and every scale predicts the horizon.

    The defaults follow the long-term forecasting scripts of Time-Series-Library.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    feedforward_size : int
        Size of the hidden layer of the feed-forward networks.
    num_layers : int
        Number of past decomposable mixing blocks.
    num_down_sampling_layers : int
        Number of scales besides the window itself.
    down_sampling_window : int
        Factor by which each scale is shorter than the next finer one.
    kernel_size : int
        Length of the moving average that extracts the trends; odd.
    dropout : float
        Dropout probability after the embedding and before the summary output layer.
    is_affine : bool
        Whether the reversible instance normalization of each scale of a forecast has a learnable affine map.

    Raises
    ------
    ValueError
        If a size is not positive, num_down_sampling_layers is negative, down_sampling_window is less than 2,
        kernel_size is not odd, or dropout is not in [0, 1).
    """

    model_size: int = 16
    feedforward_size: int = 32
    num_layers: int = 2
    num_down_sampling_layers: int = 3
    down_sampling_window: int = 2
    kernel_size: int = 25
    dropout: float = 0.1
    is_affine: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("feedforward_size", self.feedforward_size),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_down_sampling_layers < 0:
            raise ValueError(f"num_down_sampling_layers must not be negative, got {self.num_down_sampling_layers}")
        if self.down_sampling_window < 2:
            raise ValueError(f"down_sampling_window must be at least 2, got {self.down_sampling_window}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {self.kernel_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def compute_scale_lengths(self, input_length: int) -> tuple[int, ...]:
        """
        Get the number of time steps of every scale.

        Parameters
        ----------
        input_length : int
            Number of time steps of the window.

        Returns
        -------
        tuple[int, ...]
            The lengths from the finest scale, the window, to the coarsest.

        Raises
        ------
        ValueError
            If the coarsest scale would have no time step.
        """
        lengths: tuple[int, ...] = tuple(
            input_length // self.down_sampling_window**position for position in range(self.num_down_sampling_layers + 1)
        )
        if lengths[-1] < 1:
            raise ValueError(
                f"input_length {input_length} is too short for {self.num_down_sampling_layers} down-sampling layers "
                + f"of window {self.down_sampling_window}"
            )
        return lengths

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
            If the window is too short for the down-sampling layers.
        """
        return TimeMixerForecastNetwork(self, shape)

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
            If the sequences are too short for the down-sampling layers.
        """
        return TimeMixerReconstructionNetwork(self, shape)

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
            If the sequences are too short for the down-sampling layers.
        """
        return TimeMixerSummaryNetwork(self, shape)
