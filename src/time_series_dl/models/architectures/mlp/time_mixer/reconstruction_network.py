from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import TimeMixerEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerParameters


class TimeMixerReconstructionNetwork(ReconstructionNetwork):
    """
    Each channel is down-sampled into several scales and encoded independently of the other channels, and the tokens
    of the finest scale are mapped back to the values.

    Parameters
    ----------
    parameters : TimeMixerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If the coarsest scale would have no time step.
    """

    def __init__(self, parameters: TimeMixerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: TimeMixerEncoder = TimeMixerEncoder(
            parameters, shape.length, 1, is_channel_mixing=False, is_finest_output_only=True
        )
        self.projection: nn.Linear = nn.Linear(parameters.model_size, 1)

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch_size: int
        length: int
        num_channels: int
        batch_size, length, num_channels = values.shape
        scales: list[Tensor] = [
            scale.transpose(1, 2).reshape(batch_size * num_channels, -1, 1) for scale in self.encoder.downsample(values)
        ]
        reconstructed: Tensor = self.projection(self.encoder(scales)[0])
        return reconstructed.reshape(batch_size, num_channels, length).transpose(1, 2)
