from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....interfaces import MaskedReconstruction, MaskedReconstructionNetwork, SequenceShape
from .encoder import MOMENTEncoder
from .encoding import MOMENTEncoding
from .reconstruction_head import MOMENTReconstructionHead

if TYPE_CHECKING:
    from .parameters import MOMENTParameters


class MOMENTMaskedReconstructionNetwork(MaskedReconstructionNetwork):
    """
    MOMENT's masked reconstruction, as in its pretraining and its zero-shot imputation: every channel is a separate
    series, missing values are excluded from the normalization statistics, and each patch with a missing value is
    replaced by the mask embedding. With published weights it imputes zero-shot.

    The official mask is shared by the channels of a sample; here each channel has its own, which is the same when
    the channels are missing together. The network defines no loss of its own; the task's loss on the hidden values
    is the pretraining objective.

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

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return 1.0

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        batch, length, channels = values.shape
        series: Tensor = values.transpose(1, 2).flatten(end_dim=1)
        padded, is_input = self.encoder.pad(series)
        is_present, _ = self.encoder.pad(is_observed.transpose(1, 2).flatten(end_dim=1).to(values.dtype))
        is_visible: Tensor = (is_present > 0) & is_input
        encoding: MOMENTEncoding = self.encoder(padded, is_visible, is_visible, is_input)
        reconstructed: Tensor = encoding.denormalize(self.head(encoding.states))[:, -length:]
        return MaskedReconstruction(values=reconstructed.reshape(batch, channels, length).transpose(1, 2), loss=None)
