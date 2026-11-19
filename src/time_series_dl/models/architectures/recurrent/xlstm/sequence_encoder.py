from typing import override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .block_stack import XLSTMBlockStack


class XLSTMSequenceEncoder(SequenceEncoder):
    """
    Linear embedding of each step followed by a stack of xLSTM blocks; causal, so every step sees only the past.

    Parameters
    ----------
    stack : XLSTMBlockStack
        The blocks, of size ``model_size``.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    model_size : int
        Size of the feature vectors.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, stack: XLSTMBlockStack, input_length: int, input_channels: int, model_size: int) -> None:
        for name, value in (
            ("input_length", input_length),
            ("input_channels", input_channels),
            ("model_size", model_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = model_size
        self.embedding: nn.Linear = nn.Linear(input_channels, model_size)
        self.stack: XLSTMBlockStack = stack

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
        features: Tensor = self.stack(self.embedding(values))
        return features
