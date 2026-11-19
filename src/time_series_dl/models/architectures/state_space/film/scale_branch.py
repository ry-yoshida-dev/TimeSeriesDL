from typing import cast

from torch import Tensor, nn

from .frequency_enhanced_layer import FiLMFrequencyEnhancedLayer
from .legendre_projection import FiLMLegendreProjection


class FiLMScaleBranch(nn.Module):
    """
    One scale of FiLM: the Legendre memory of a window, its frequency enhanced layer, and the reconstruction of the
    horizon from the last filtered memory state.

    Parameters
    ----------
    state_size : int
        Number of Legendre polynomials.
    window_length : int
        Number of steps of the window the memory covers; at least the horizon.
    input_length : int
        Number of time steps of the input.
    horizon : int
        Number of future steps.
    num_modes : int
        Largest number of lowest Fourier modes kept.
    """

    def __init__(self, state_size: int, window_length: int, input_length: int, horizon: int, num_modes: int) -> None:
        super().__init__()
        self.projection: FiLMLegendreProjection = FiLMLegendreProjection(
            state_size, window_length, input_length, horizon
        )
        self.layer: FiLMFrequencyEnhancedLayer = FiLMFrequencyEnhancedLayer(
            state_size, self.projection.read_length, num_modes
        )

    def forward(self, values: Tensor) -> Tensor:
        """
        Forecast the horizon of each series at this scale.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, channels, input_length).

        Returns
        -------
        Tensor
            Forecasts with shape (batch, channels, horizon).
        """
        filtered: Tensor = self.layer(self.projection.encode(values))
        return self.projection.reconstruct(filtered[..., -1])

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
