from dataclasses import dataclass, field

from ....components.layers import TemporalPooling
from ....components.masked_modeling import GeometricMasking, Masking
from ....interfaces import SequenceShape
from ....interfaces.protocols.encoding import EncodingModel
from ...transformer.vanilla import TransformerParameters
from .pretraining_network import SimMTMPretrainingNetwork


def _build_default_masking() -> Masking:
    return GeometricMasking(ratio=0.5, mean_length=3.0)


@dataclass(frozen=True, kw_only=True)
class SimMTMParameters:
    """
    SimMTM (Dong et al., NeurIPS 2023): masked time-series modeling that reconstructs each sequence from the
    similarity-weighted point-wise features of several masked versions of it and of the other sequences, around the
    encoder of any model.

    Defaults follow the official code (geometric masks hiding half of the values in segments of mean length 3,
    three masked versions, temperature 0.2, learned loss weights), with the official Transformer encoder of
    ``TransformerParameters``. The series-wise representation averages the features over time by default.

    Attributes
    ----------
    encoder : EncodingModel
        Parameters of the model whose encoder is pretrained, such as ``TSTParameters`` or ``S4Parameters``.
    masking : Masking
        The random masks of the masked versions.
    num_masked : int
        Number of masked versions of each sequence.
    temperature : float
        Temperature of the series-wise similarities.
    projection_size : int
        Size of the series-wise representation.
    pooling : TemporalPooling
        Reduction of the point-wise features before the series-wise projector, and the instance-level
        representation; ``FLATTEN`` as in the official code is costly for long sequences or large encoders.
    constraint_weight : float | None
        Fixed weight of the constraint loss added to the reconstruction loss (lambda of the paper); learned weights
        of both losses as in the official code when None.

    Raises
    ------
    ValueError
        If a count or size is not positive, temperature is not positive, or constraint_weight is negative.
    """

    encoder: EncodingModel = field(default_factory=TransformerParameters)
    masking: Masking = field(default_factory=_build_default_masking)
    num_masked: int = 3
    temperature: float = 0.2
    projection_size: int = 128
    pooling: TemporalPooling = TemporalPooling.MEAN
    constraint_weight: float | None = None

    def __post_init__(self) -> None:
        if self.num_masked <= 0:
            raise ValueError(f"num_masked must be positive, got {self.num_masked}")
        if self.projection_size <= 0:
            raise ValueError(f"projection_size must be positive, got {self.projection_size}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.constraint_weight is not None and self.constraint_weight < 0.0:
            raise ValueError(f"constraint_weight must not be negative, got {self.constraint_weight}")

    def build_pretraining_network(self, shape: SequenceShape) -> SimMTMPretrainingNetwork:
        """
        Build an untrained pretraining network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the unlabeled sequences.

        Returns
        -------
        SimMTMPretrainingNetwork
            The network.
        """
        return SimMTMPretrainingNetwork(
            self.encoder.build_encoder(shape.length, shape.channels),
            shape,
            self.masking,
            self.num_masked,
            self.temperature,
            self.projection_size,
            self.pooling,
            self.constraint_weight,
        )
