from typing import override

from torch import Tensor

from ....components.layers import TemporalPooling
from ....components.masked_modeling import Masking, SequenceDecoder
from ....interfaces import SequenceEncoder, SequenceShape
from ....interfaces.pretraining_network import PretrainingNetwork


class MaskedReconstructionPretrainingNetwork(PretrainingNetwork):
    """
    Masked-value pretraining of TST and of the self-supervised PatchTST around any sequence encoder: hidden values
    are set to zero, and the encoder's features are decoded linearly back to the values, with the mean squared error
    of the hidden values as the loss.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoder to pretrain.
    shape : SequenceShape
        The dimensions of the unlabeled sequences.
    masking : Masking
        The random masks of hidden values.
    pooling : TemporalPooling
        Reduction of the features to the instance-level representation.
    """

    def __init__(
        self, encoder: SequenceEncoder, shape: SequenceShape, masking: Masking, pooling: TemporalPooling
    ) -> None:
        super().__init__()
        self.sequence_encoder: SequenceEncoder = encoder
        self.masking: Masking = masking
        self.feature_pooling: TemporalPooling = pooling
        self.decoder: SequenceDecoder = SequenceDecoder(
            encoder.output_length, encoder.output_size, shape.length, shape.channels
        )

    @property
    @override
    def encoder(self) -> SequenceEncoder:
        return self.sequence_encoder

    @property
    @override
    def pooling(self) -> TemporalPooling:
        return self.feature_pooling

    @override
    def forward(self, values: Tensor) -> Tensor:
        is_hidden: Tensor = self.masking.sample(values.shape[0], values.shape[1], values.shape[2], values.device)
        return self.compute_loss(values, is_hidden)

    def compute_loss(self, values: Tensor, is_hidden: Tensor) -> Tensor:
        """
        Compute the loss of reconstructing given hidden values.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).
        is_hidden : Tensor
            Booleans of the same shape, True for each value hidden from the encoder.

        Returns
        -------
        Tensor
            The mean squared error over the hidden values; zero when nothing is hidden.
        """
        reconstructed: Tensor = self.decoder(self.sequence_encoder(values.masked_fill(is_hidden, 0.0)))
        squared_errors: Tensor = (reconstructed - values).square() * is_hidden
        return squared_errors.sum() / is_hidden.sum().clamp(min=1)
