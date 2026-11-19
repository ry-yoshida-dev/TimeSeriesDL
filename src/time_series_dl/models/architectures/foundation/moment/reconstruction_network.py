from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....interfaces import ReconstructionNetwork, SequenceShape
from .encoder import MOMENTEncoder
from .encoding import MOMENTEncoding
from .reconstruction_head import MOMENTReconstructionHead

if TYPE_CHECKING:
    from .parameters import MOMENTParameters


class MOMENTReconstructionNetwork(ReconstructionNetwork):
    """
    MOMENT's reconstruction of every channel as a separate series with nothing hidden, as the official anomaly
    detection does; with published weights it reconstructs zero-shot.

    Sequences shorter than ``input_length`` are left-padded; as in the official default, a patch that is partly
    padding is hidden and excluded from the normalization statistics.

    Parameters
    ----------
    parameters : MOMENTParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If the sequences are longer than ``input_length``.
    """

    def __init__(self, parameters: MOMENTParameters, shape: SequenceShape) -> None:
        if shape.length > parameters.input_length:
            raise ValueError(f"MOMENT reads at most {parameters.input_length} steps, got {shape.length}")
        super().__init__()
        self.encoder: MOMENTEncoder = MOMENTEncoder(parameters)
        self.head: MOMENTReconstructionHead = MOMENTReconstructionHead(
            parameters.model_size, parameters.patch_length, parameters.head_dropout
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch, length, channels = values.shape
        series: Tensor = values.transpose(1, 2).flatten(end_dim=1)
        padded, is_input = self.encoder.pad(series)
        is_whole: Tensor = self.encoder.expand_whole_patches(is_input)
        encoding: MOMENTEncoding = self.encoder(padded, is_whole, is_whole, is_input)
        reconstructed: Tensor = encoding.denormalize(self.head(encoding.states))[:, -length:]
        return reconstructed.reshape(batch, channels, length).transpose(1, 2)
