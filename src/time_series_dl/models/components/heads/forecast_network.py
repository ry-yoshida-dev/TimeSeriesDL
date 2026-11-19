from typing import override

from torch import Tensor, nn

from ...interfaces import ForecastNetwork, ForecastShape, SequenceEncoder


class EncodedForecastNetwork(ForecastNetwork):
    """
    Encoder followed by a linear map from all flattened features to the whole horizon.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoder of the past values.
    shape : ForecastShape
        The dimensions of the forecast.
    dropout : float
        Dropout probability before the output layer.
    """

    def __init__(self, encoder: SequenceEncoder, shape: ForecastShape, dropout: float) -> None:
        super().__init__()
        self.encoder: SequenceEncoder = encoder
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.projection: nn.Linear = nn.Linear(
            encoder.output_length * encoder.output_size, shape.horizon * shape.output_channels * shape.num_parameters
        )

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past)
        outputs: Tensor = self.projection(self.dropout(features.flatten(start_dim=1)))
        return outputs.reshape(-1, self.horizon, self.output_channels, self.num_parameters)
