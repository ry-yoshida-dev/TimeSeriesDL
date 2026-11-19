from __future__ import annotations

from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .block_kind import NBEATSBlockKind
from .forecast_network import NBEATSForecastNetwork
from .stack import NBEATSStack


@dataclass(frozen=True, kw_only=True)
class NBEATSParameters:
    """
    N-BEATS (Oreshkin et al., 2020): doubly residual stacks of fully connected blocks with basis expansion.

    The network is univariate: it forecasts each target channel from that channel's own history, with weights shared
    across channels, and ignores the channels that are not forecast. The defaults are the generic architecture of
    the paper (30 stacks of one block with 4 layers of width 512); ``interpretable`` builds the trend and seasonality
    architecture.

    Attributes
    ----------
    stacks : tuple[NBEATSStack, ...]
        The stacks in order.
    polynomial_degree : int
        Highest power of the polynomial basis of trend blocks (3 in the paper's configuration).
    num_harmonics : int
        Number of Fourier frequencies per cycle of the horizon in seasonality blocks (1 in the paper's
        configuration).

    Raises
    ------
    ValueError
        If there is no stack, the degree is negative or the number of harmonics is not positive.
    """

    stacks: tuple[NBEATSStack, ...] = (NBEATSStack(),) * 30
    polynomial_degree: int = 3
    num_harmonics: int = 1

    def __post_init__(self) -> None:
        if not self.stacks:
            raise ValueError("stacks must not be empty.")
        if self.polynomial_degree < 0:
            raise ValueError(f"polynomial_degree must not be negative, got {self.polynomial_degree}")
        if self.num_harmonics <= 0:
            raise ValueError(f"num_harmonics must be positive, got {self.num_harmonics}")

    @classmethod
    def generic(cls, num_stacks: int = 30, num_layers: int = 4, hidden_size: int = 512) -> NBEATSParameters:
        """
        Build the generic architecture: stacks of one block with the identity basis.

        Parameters
        ----------
        num_stacks : int
            Number of stacks.
        num_layers : int
            Number of hidden layers of each block.
        hidden_size : int
            Width of the hidden layers.

        Returns
        -------
        NBEATSParameters
            The parameters.

        Raises
        ------
        ValueError
            If a count or size is not positive.
        """
        if num_stacks <= 0:
            raise ValueError(f"num_stacks must be positive, got {num_stacks}")
        stack: NBEATSStack = NBEATSStack(
            kind=NBEATSBlockKind.GENERIC, num_blocks=1, num_layers=num_layers, hidden_size=hidden_size
        )
        return cls(stacks=(stack,) * num_stacks)

    @classmethod
    def interpretable(
        cls,
        num_blocks: int = 3,
        num_layers: int = 4,
        trend_hidden_size: int = 256,
        seasonality_hidden_size: int = 2048,
        polynomial_degree: int = 3,
        num_harmonics: int = 1,
    ) -> NBEATSParameters:
        """
        Build the interpretable architecture: a trend stack followed by a seasonality stack, each sharing its weights.

        Parameters
        ----------
        num_blocks : int
            Number of blocks of each stack.
        num_layers : int
            Number of hidden layers of each block.
        trend_hidden_size : int
            Width of the hidden layers of the trend stack.
        seasonality_hidden_size : int
            Width of the hidden layers of the seasonality stack.
        polynomial_degree : int
            Highest power of the trend basis.
        num_harmonics : int
            Number of Fourier frequencies per cycle of the horizon.

        Returns
        -------
        NBEATSParameters
            The parameters.

        Raises
        ------
        ValueError
            If a count or size is not positive or the degree is negative.
        """
        trend: NBEATSStack = NBEATSStack(
            kind=NBEATSBlockKind.TREND,
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_size=trend_hidden_size,
            is_weight_shared=True,
        )
        seasonality: NBEATSStack = NBEATSStack(
            kind=NBEATSBlockKind.SEASONALITY,
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_size=seasonality_hidden_size,
            is_weight_shared=True,
        )
        return cls(stacks=(trend, seasonality), polynomial_degree=polynomial_degree, num_harmonics=num_harmonics)

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
        return NBEATSForecastNetwork(self, shape)
