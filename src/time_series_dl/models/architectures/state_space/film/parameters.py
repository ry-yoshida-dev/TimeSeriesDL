from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import FiLMForecastNetwork


@dataclass(frozen=True, kw_only=True)
class FiLMParameters:
    """
    FiLM (Zhou et al., 2022): Frequency improved Legendre Memory, which projects each channel onto Legendre
    polynomials with a fixed HiPPO-LegT memory over windows of several multiples of the horizon, filters the memory in
    the frequency domain and reconstructs the forecast from it.

    Channels are forecast independently with shared weights; the defaults follow the official scripts.

    Attributes
    ----------
    state_size : int
        Number of Legendre polynomials of each memory.
    scales : tuple[int, ...]
        Window lengths as multiples of the horizon, one memory per scale.
    num_modes : int
        Largest number of lowest Fourier modes kept by the frequency enhanced layers.

    Raises
    ------
    ValueError
        If a size is not positive, or scales is empty or holds a non-positive or repeated scale.
    """

    state_size: int = 256
    scales: tuple[int, ...] = (1, 2, 4)
    num_modes: int = 32

    def __post_init__(self) -> None:
        if self.state_size <= 0:
            raise ValueError(f"state_size must be positive, got {self.state_size}")
        if self.num_modes <= 0:
            raise ValueError(f"num_modes must be positive, got {self.num_modes}")
        if not self.scales or any(scale <= 0 for scale in self.scales):
            raise ValueError(f"scales must be positive and not empty, got {self.scales}")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError(f"scales must not repeat, got {self.scales}")

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
        return FiLMForecastNetwork(self, shape)
