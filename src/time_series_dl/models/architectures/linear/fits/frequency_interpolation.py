import math
from typing import cast

import torch
from torch import Tensor, nn


class FITSFrequencyInterpolation(nn.Module):
    """
    Low-pass filter of the spectrum of each channel along time, complex-valued linear interpolation to the spectrum
    of a longer (or equally long) series, and the inverse transform scaled by the length ratio.

    Parameters
    ----------
    input_length : int
        Number of input time steps.
    output_length : int
        Number of output time steps; at least ``input_length``.
    cutoff_frequency : int
        Number of lowest frequencies kept; capped at the number of frequencies of the input.
    channels : int
        Number of channels.
    is_individual : bool
        Whether each channel has its own complex weights.
    num_outputs : int
        Number of output series per channel, each with its own complex output frequencies.

    Raises
    ------
    ValueError
        If a size is not positive or output_length is shorter than input_length.
    """

    def __init__(
        self,
        input_length: int,
        output_length: int,
        cutoff_frequency: int,
        channels: int,
        is_individual: bool,
        num_outputs: int,
    ) -> None:
        for name, value in (
            ("input_length", input_length),
            ("cutoff_frequency", cutoff_frequency),
            ("channels", channels),
            ("num_outputs", num_outputs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if output_length < input_length:
            raise ValueError(f"output_length must be at least input_length {input_length}, got {output_length}")
        super().__init__()
        self.output_length: int = output_length
        self.num_outputs: int = num_outputs
        self.length_ratio: float = output_length / input_length
        self.num_input_frequencies: int = min(cutoff_frequency, input_length // 2 + 1)
        self.num_output_frequencies: int = min(
            int(self.num_input_frequencies * self.length_ratio), output_length // 2 + 1
        )
        count: int = channels if is_individual else 1
        bound: float = 1.0 / math.sqrt(self.num_input_frequencies)
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(count, self.num_input_frequencies, num_outputs * self.num_output_frequencies, 2).uniform_(
                -bound, bound
            )
        )
        self.bias: nn.Parameter = nn.Parameter(
            torch.empty(count, num_outputs * self.num_output_frequencies, 2).uniform_(-bound, bound)
        )

    def forward(self, values: Tensor) -> Tensor:
        """
        Interpolate a batch of series in the frequency domain.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Values with shape (batch, output_length, channels, num_outputs).
        """
        channels: int = values.shape[2]
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(values, dim=1))[:, : self.num_input_frequencies]
        weight: Tensor = torch.view_as_complex(self.weight).expand(channels, -1, -1)
        bias: Tensor = torch.view_as_complex(self.bias).expand(channels, -1)
        mixed: Tensor = torch.einsum("bfc,cfg->bcg", spectrum, weight) + bias
        mixed = mixed.reshape(*mixed.shape[:2], self.num_outputs, self.num_output_frequencies)
        padding: Tensor = mixed.new_zeros(*mixed.shape[:3], self.output_length // 2 + 1 - self.num_output_frequencies)
        series: Tensor = cast(
            Tensor, torch.fft.irfft(torch.cat([mixed, padding], dim=-1), n=self.output_length, dim=-1)
        )
        return (series * self.length_ratio).permute(0, 3, 1, 2)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
