import math
from typing import cast

import torch
from torch import Tensor, nn

from .dynamic_mode_decomposition import KoopaDynamicModeDecomposition


class KoopaTimeVariantPredictor(nn.Module):
    """
    Time-variant Koopman predictor: the window is cut into segments of every channel, each segment is encoded to a
    state, an operator fitted to the window's states reconstructs them and rolls them forward, and the states are
    decoded back to segments.

    The window is padded in front with its own last values to a whole number of segments, as in Time-Series-Library.

    Parameters
    ----------
    encoder : nn.Module
        Map from a segment of every channel to its state; shared with the other blocks.
    decoder : nn.Module
        Map from a state to a segment of every channel; shared with the other blocks.
    segment_length : int
        Number of time steps of each segment.
    input_length : int
        Number of time steps of the window.
    horizon : int
        Number of time steps forecast.

    Raises
    ------
    ValueError
        If the window holds fewer than 2 segments.
    """

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, segment_length: int, input_length: int, horizon: int
    ) -> None:
        num_input_segments: int = math.ceil(input_length / segment_length)
        if num_input_segments < 2:
            raise ValueError(
                f"the window of {input_length} steps must hold at least 2 segments of {segment_length} steps"
            )
        super().__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.dynamics: KoopaDynamicModeDecomposition = KoopaDynamicModeDecomposition()
        self.input_length: int = input_length
        self.horizon: int = horizon
        self.num_input_segments: int = num_input_segments
        self.num_output_segments: int = math.ceil(horizon / segment_length)
        self.padding_length: int = num_input_segments * segment_length - input_length

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Reconstruct and forecast a batch of series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            The reconstruction with shape (batch, input_length, channels) and the forecasts with shape
            (batch, horizon, channels).
        """
        batch_size: int = values.shape[0]
        channels: int = values.shape[2]
        padded: Tensor = torch.cat([values[:, values.shape[1] - self.padding_length :], values], dim=1)
        states: Tensor = self.encoder(padded.reshape(batch_size, self.num_input_segments, -1))
        reconstructed_states: Tensor
        predicted_states: Tensor
        reconstructed_states, predicted_states = self.dynamics(states, self.num_output_segments)
        reconstruction: Tensor = self.decoder(reconstructed_states).reshape(batch_size, -1, channels)
        forecasts: Tensor = self.decoder(predicted_states).reshape(batch_size, -1, channels)
        return reconstruction[:, self.padding_length :], forecasts[:, : self.horizon]

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values))
