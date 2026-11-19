from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import TimeMixerEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerParameters


class TimeMixerSummaryNetwork(SummaryNetwork):
    """
    Every channel is embedded into shared tokens at several scales, and the encoded tokens of the finest scale pass a
    GELU, dropout and a linear map of all flattened tokens to the output vector.

    Parameters
    ----------
    parameters : TimeMixerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.

    Raises
    ------
    ValueError
        If the coarsest scale would have no time step.
    """

    def __init__(self, parameters: TimeMixerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: TimeMixerEncoder = TimeMixerEncoder(
            parameters, shape.input_length, shape.input_channels, is_channel_mixing=True, is_finest_output_only=True
        )
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size * shape.input_length, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(self.encoder.downsample(values))[0]
        outputs: Tensor = self.projection(self.dropout(self.activation(tokens)).flatten(start_dim=1))
        return outputs
