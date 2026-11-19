from collections.abc import Callable
from typing import override

from torch import Tensor, nn

from ....components.state_space import StateSpaceLayer
from ....interfaces import SequenceEncoder


class DeepStateSpaceSequenceEncoder(SequenceEncoder):
    """
    Linear embedding followed by residual blocks x + dropout(layer(norm(x))) around state space layers, and a final
    normalization.

    Parameters
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    model_size : int
        Size of the feature vectors.
    num_layers : int
        Number of residual blocks.
    dropout : float
        Dropout probability of each block's output.
    is_bidirectional : bool
        Whether each block adds a second layer that reads the sequence backward, so that every step sees the whole
        sequence.
    build_layer : Callable[[], StateSpaceLayer]
        Builds one untrained state space layer of size ``model_size``.
    build_normalization : Callable[[], nn.Module]
        Builds one normalization over the last dimension of size ``model_size``.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(
        self,
        input_length: int,
        input_channels: int,
        model_size: int,
        num_layers: int,
        dropout: float,
        is_bidirectional: bool,
        build_layer: Callable[[], StateSpaceLayer],
        build_normalization: Callable[[], nn.Module],
    ) -> None:
        for name, value in (
            ("input_length", input_length),
            ("input_channels", input_channels),
            ("model_size", model_size),
            ("num_layers", num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = model_size
        self.embedding: nn.Linear = nn.Linear(input_channels, model_size)
        self.normalizations: nn.ModuleList = nn.ModuleList(build_normalization() for _ in range(num_layers))
        self.forward_layers: nn.ModuleList = nn.ModuleList(build_layer() for _ in range(num_layers))
        self.backward_layers: nn.ModuleList = nn.ModuleList(
            build_layer() for _ in range(num_layers if is_bidirectional else 0)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.final_normalization: nn.Module = build_normalization()

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
        for index, (normalization, layer) in enumerate(zip(self.normalizations, self.forward_layers, strict=True)):
            normalized: Tensor = normalization(features)
            mixed: Tensor = layer(normalized)
            if self.backward_layers:
                mixed = mixed + self.backward_layers[index](normalized.flip(1)).flip(1)
            features = features + self.dropout(mixed)
        normalized_features: Tensor = self.final_normalization(features)
        return normalized_features
