from dataclasses import dataclass

from torch import nn

from ....components.layers import TemporalPooling
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import TSMixerForecastNetwork
from .mixer_block import TSMixerBlock
from .normalization import TSMixerNormalization
from .reconstruction_network import TSMixerReconstructionNetwork
from .summary_network import TSMixerSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class TSMixerParameters:
    """
    TSMixer (Chen et al., 2023): stacked time-mixing and feature-mixing MLPs, then a linear map along time.

    The defaults follow the smaller configurations of the paper's long-term forecasting experiments (2 blocks, feature
    MLP of 64 units, batch normalization); the paper tunes the dropout per dataset.

    Attributes
    ----------
    num_blocks : int
        Number of mixer blocks.
    hidden_size : int
        Size of the hidden layer of each feature-mixing MLP.
    dropout : float
        Dropout probability after each mixing layer.
    normalization : TSMixerNormalization
        Normalization in front of each mixing MLP.
    pooling : TemporalPooling
        Reduction of the mixed sequence for summary networks.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    num_blocks: int = 2
    hidden_size: int = 64
    dropout: float = 0.1
    normalization: TSMixerNormalization = TSMixerNormalization.BATCH
    pooling: TemporalPooling = TemporalPooling.FLATTEN

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_mixer(self, length: int, channels: int) -> nn.Sequential:
        """
        Build the untrained mixer blocks, which keep the shape (batch, length, channels) of their input.

        Parameters
        ----------
        length : int
            Number of time steps.
        channels : int
            Number of channels.

        Returns
        -------
        nn.Sequential
            The blocks.
        """
        return nn.Sequential(
            *(
                TSMixerBlock(length, channels, self.hidden_size, self.dropout, self.normalization)
                for _ in range(self.num_blocks)
            )
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
        return TSMixerForecastNetwork(self, shape)

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
        return TSMixerReconstructionNetwork(self, shape)

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
        return TSMixerSummaryNetwork(self, shape)
