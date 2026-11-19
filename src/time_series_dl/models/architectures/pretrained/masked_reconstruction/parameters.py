from dataclasses import dataclass, field

from ....components.layers import TemporalPooling
from ....components.masked_modeling import GeometricMasking, Masking
from ....interfaces import SequenceShape
from ....interfaces.protocols.encoding import EncodingModel
from ...transformer.tst import TSTParameters
from .pretraining_network import MaskedReconstructionPretrainingNetwork


@dataclass(frozen=True, kw_only=True)
class MaskedReconstructionParameters:
    """
    Masked-value reconstruction pretraining (Zerveas et al., KDD 2021; Nie et al., ICLR 2023) around the encoder of
    any model.

    The defaults are those of TST: its encoder and geometric masks hiding 15% of the values of each channel in
    segments of mean length 3. ``PatchMasking()`` gives the masking of the self-supervised PatchTST (40% of the
    non-overlapping patches of 12 steps).

    Attributes
    ----------
    encoder : EncodingModel
        Parameters of the model whose encoder is pretrained, such as ``TSTParameters``.
    masking : Masking
        The random masks of hidden values.
    pooling : TemporalPooling
        Reduction of the features to the instance-level representation; flattening, as the output layers of TST.
    """

    encoder: EncodingModel = field(default_factory=TSTParameters)
    masking: Masking = field(default_factory=GeometricMasking)
    pooling: TemporalPooling = TemporalPooling.FLATTEN

    def build_pretraining_network(self, shape: SequenceShape) -> MaskedReconstructionPretrainingNetwork:
        """
        Build an untrained pretraining network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the unlabeled sequences.

        Returns
        -------
        MaskedReconstructionPretrainingNetwork
            The network.
        """
        return MaskedReconstructionPretrainingNetwork(
            self.encoder.build_encoder(shape.length, shape.channels), shape, self.masking, self.pooling
        )
