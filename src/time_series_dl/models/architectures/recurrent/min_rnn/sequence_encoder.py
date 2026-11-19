from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.layers import Activation, MultilayerPerceptron
from ....interfaces import SequenceEncoder
from .minimal_recurrent_layer import MinimalRecurrentLayer

if TYPE_CHECKING:
    from .parameters import MinRNNParameters


class MinRNNSequenceEncoder(SequenceEncoder):
    """
    Linear embedding followed by residual blocks of a minimal recurrent layer and a feedforward network, each applied
    to the layer-normalized features, and a final layer normalization; causal, so every step sees only the past.

    Parameters
    ----------
    parameters : MinRNNParameters
        The layer configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, parameters: MinRNNParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(f"input_length and input_channels must be positive, got {input_length}, {input_channels}")
        super().__init__()
        size: int = parameters.model_size
        self._output_length: int = input_length
        self._output_size: int = size
        self.embedding: nn.Linear = nn.Linear(input_channels, size)
        self.recurrent_normalizations: nn.ModuleList = nn.ModuleList(
            nn.LayerNorm(size) for _ in range(parameters.num_layers)
        )
        self.recurrent_layers: nn.ModuleList = nn.ModuleList(
            MinimalRecurrentLayer(size, parameters.hidden_size, parameters.cell) for _ in range(parameters.num_layers)
        )
        self.feedforward_normalizations: nn.ModuleList = nn.ModuleList(
            nn.LayerNorm(size) for _ in range(parameters.num_layers)
        )
        self.feedforwards: nn.ModuleList = nn.ModuleList(
            MultilayerPerceptron(size, [parameters.feedforward_factor * size], size, Activation.GELU)
            for _ in range(parameters.num_layers)
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.final_normalization: nn.LayerNorm = nn.LayerNorm(size)

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.embedding(values)
        for recurrent_normalization, recurrent_layer, feedforward_normalization, feedforward in zip(
            self.recurrent_normalizations,
            self.recurrent_layers,
            self.feedforward_normalizations,
            self.feedforwards,
            strict=True,
        ):
            features = features + self.dropout(recurrent_layer(recurrent_normalization(features)))
            features = features + self.dropout(feedforward(feedforward_normalization(features)))
        normalized: Tensor = self.final_normalization(features)
        return normalized
