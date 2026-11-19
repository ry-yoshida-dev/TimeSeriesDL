from typing import Self, override

import torch
from torch import Tensor

from ....interfaces import SequenceEncoder


class FrozenSequenceEncoder(SequenceEncoder):
    """
    Encoder whose weights are not trained, for linear probing: its parameters need no gradient, and it stays in
    evaluation mode, so dropout is off and batch normalization keeps its pretrained statistics.

    Parameters
    ----------
    encoder : SequenceEncoder
        The pretrained encoder; its parameters are frozen in place.
    """

    def __init__(self, encoder: SequenceEncoder) -> None:
        super().__init__()
        self.encoder: SequenceEncoder = encoder
        _ = self.encoder.requires_grad_(False)
        _ = self.encoder.eval()

    @property
    @override
    def output_length(self) -> int:
        return self.encoder.output_length

    @property
    @override
    def output_size(self) -> int:
        return self.encoder.output_size

    @override
    def train(self, mode: bool = True) -> Self:
        _ = super().train(mode)
        _ = self.encoder.eval()
        return self

    @override
    def forward(self, values: Tensor) -> Tensor:
        with torch.no_grad():
            features: Tensor = self.encoder(values)
        return features
