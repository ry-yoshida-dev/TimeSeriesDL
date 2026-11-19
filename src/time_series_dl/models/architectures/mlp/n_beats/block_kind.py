import math
from enum import Enum

import torch
from torch import Tensor


class NBEATSBlockKind(Enum):
    """
    Basis that an N-BEATS block expands its coefficients with.

    Attributes
    ----------
    GENERIC
        Identity basis: the coefficients are the backcast and forecast values themselves.
    TREND
        Polynomial basis of a low degree over time scaled to [0, 1), for a slowly varying trend.
    SEASONALITY
        Fourier basis of the harmonics of the horizon, for periodic patterns.
    """

    GENERIC = "generic"
    TREND = "trend"
    SEASONALITY = "seasonality"

    def build_bases(
        self, input_length: int, horizon: int, polynomial_degree: int, num_harmonics: int
    ) -> tuple[Tensor, Tensor]:
        """
        Build the backcast and forecast bases.

        Parameters
        ----------
        input_length : int
            Number of past time steps.
        horizon : int
            Number of future time steps.
        polynomial_degree : int
            Highest power of the trend basis.
        num_harmonics : int
            Number of Fourier frequencies per cycle of the horizon (the frequency resolution is ``1 / num_harmonics``).

        Returns
        -------
        tuple[Tensor, Tensor]
            The backcast basis with shape (backcast coefficients, input_length) and the forecast basis with shape
            (forecast coefficients, horizon).

        Raises
        ------
        ValueError
            If a length is not positive, the degree is negative or the number of harmonics is not positive.
        """
        if input_length <= 0 or horizon <= 0:
            raise ValueError(f"input_length and horizon must be positive, got {input_length} and {horizon}")
        if polynomial_degree < 0:
            raise ValueError(f"polynomial_degree must not be negative, got {polynomial_degree}")
        if num_harmonics <= 0:
            raise ValueError(f"num_harmonics must be positive, got {num_harmonics}")
        match self:
            case NBEATSBlockKind.GENERIC:
                return torch.eye(input_length), torch.eye(horizon)
            case NBEATSBlockKind.TREND:
                powers: Tensor = torch.arange(polynomial_degree + 1, dtype=torch.float32).unsqueeze(1)
                backcast_time: Tensor = torch.arange(input_length, dtype=torch.float32) / input_length
                forecast_time: Tensor = torch.arange(horizon, dtype=torch.float32) / horizon
                return backcast_time.pow(powers), forecast_time.pow(powers)
            case NBEATSBlockKind.SEASONALITY:
                # Frequencies in cycles per horizon, below the Nyquist frequency of the horizon.
                count: int = max(math.ceil(num_harmonics * horizon / 2), 1)
                frequencies: Tensor = torch.arange(count, dtype=torch.float32).unsqueeze(1) / num_harmonics
                backcast_time = (torch.arange(input_length, dtype=torch.float32) - input_length) / horizon
                forecast_time = torch.arange(horizon, dtype=torch.float32) / horizon
                return (
                    self._fourier(frequencies, backcast_time),
                    self._fourier(frequencies, forecast_time),
                )

    @staticmethod
    def _fourier(frequencies: Tensor, time: Tensor) -> Tensor:
        angles: Tensor = 2.0 * math.pi * frequencies * time
        return torch.cat([torch.cos(angles), torch.sin(angles[1:])], dim=0)
