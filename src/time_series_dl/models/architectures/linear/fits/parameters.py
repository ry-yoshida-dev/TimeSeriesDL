from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape, ReconstructionNetwork, SequenceShape
from .forecast_network import FITSForecastNetwork
from .reconstruction_network import FITSReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class FITSParameters:
    """
    FITS (Xu et al., 2024): Frequency Interpolation Time Series analysis, a low-pass filter followed by a
    complex-valued linear interpolation of the spectrum.

    The cutoff frequency keeps ``harmonic_order`` harmonics of the base period, by the official formula
    ``(input_length // base_period + 1) * harmonic_order + 10``. The defaults are the argument defaults of the
    official code; its scripts tune the harmonic order per dataset.

    Attributes
    ----------
    base_period : int
        Length of the dominant period of the data in time steps (24 for hourly data with a daily cycle).
    harmonic_order : int
        Number of harmonics of the base period kept by the low-pass filter.
    is_individual : bool
        Whether each channel has its own complex weights.

    Raises
    ------
    ValueError
        If base_period or harmonic_order is not positive.
    """

    base_period: int = 24
    harmonic_order: int = 2
    is_individual: bool = False

    def __post_init__(self) -> None:
        if self.base_period <= 0:
            raise ValueError(f"base_period must be positive, got {self.base_period}")
        if self.harmonic_order <= 0:
            raise ValueError(f"harmonic_order must be positive, got {self.harmonic_order}")

    def compute_cutoff_frequency(self, input_length: int) -> int:
        """
        Compute the number of lowest frequencies kept by the low-pass filter, before capping at the number of
        frequencies of the window.

        Parameters
        ----------
        input_length : int
            Number of time steps of the window.

        Returns
        -------
        int
            The cutoff frequency.
        """
        return (input_length // self.base_period + 1) * self.harmonic_order + 10

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
        return FITSForecastNetwork(self, shape)

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
        return FITSReconstructionNetwork(self, shape)
