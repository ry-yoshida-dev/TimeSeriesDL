from dataclasses import dataclass, field

from ....components.irregular_heads import IrregularEncoderHead
from ....interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....interfaces.shapes.irregular_sequence import IrregularSequenceShape
from ....interfaces.shapes.irregular_summary import IrregularSummaryShape
from .irregular_interpolation_network import MTANInterpolationNetwork
from .irregular_sequence_encoder import MTANSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class MTANParameters:
    """
    mTAN (Shukla and Marlin, 2021): multi-time attention networks for irregularly sampled sequences. Classification
    and regression use the mTAND-Enc encoder with the output layers of ``head``; interpolation and forecasting use
    the variational mTAND-Full encoder-decoder.

    Defaults follow the authors' PhysioNet scripts: those of classification for the shared fields, and those of
    interpolation for the latent state.

    Attributes
    ----------
    hidden_size : int
        Size of the attention outputs and of the GRU states.
    time_embedding_size : int
        Size of the learned time embeddings.
    num_heads : int
        Number of attention heads.
    num_reference_points : int
        Number of equally spaced reference times in [0, 1], the scaled range of the training times.
    latent_size : int
        Size of each reference latent state of the interpolation network.
    projection_size : int
        Hidden size of the layers that output the posterior and the values of the interpolation network.
    observation_std : float
        Standard deviation of the Gaussian likelihood of the interpolation network.
    kl_weight : float
        Weight of the Kullback-Leibler divergence in the loss of the interpolation network.
    head : IrregularEncoderHead
        Output layers of the summary network.

    Raises
    ------
    ValueError
        If a size is not positive, time_embedding_size is less than 2 or not divisible by num_heads,
        observation_std is not positive or kl_weight is negative.
    """

    hidden_size: int = 256
    time_embedding_size: int = 128
    num_heads: int = 1
    num_reference_points: int = 128
    latent_size: int = 16
    projection_size: int = 50
    observation_std: float = 0.01
    kl_weight: float = 1.0
    head: IrregularEncoderHead = field(default_factory=lambda: IrregularEncoderHead(hidden_sizes=(300, 300)))

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("num_reference_points", self.num_reference_points),
            ("latent_size", self.latent_size),
            ("projection_size", self.projection_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.time_embedding_size < 2:
            raise ValueError(f"time_embedding_size must be at least 2, got {self.time_embedding_size}")
        if self.time_embedding_size % self.num_heads != 0:
            raise ValueError(
                f"time_embedding_size {self.time_embedding_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.observation_std <= 0.0:
            raise ValueError(f"observation_std must be positive, got {self.observation_std}")
        if self.kl_weight < 0.0:
            raise ValueError(f"kl_weight must not be negative, got {self.kl_weight}")

    def build_encoder(self, input_channels: int) -> MTANSequenceEncoder:
        """
        Build the untrained mTAND-Enc encoder.

        Parameters
        ----------
        input_channels : int
            Number of channels of the sequences.

        Returns
        -------
        MTANSequenceEncoder
            The encoder.
        """
        return MTANSequenceEncoder(
            input_channels, self.hidden_size, self.time_embedding_size, self.num_heads, self.num_reference_points
        )

    def build_irregular_summary_network(self, shape: IrregularSummaryShape) -> IrregularSummaryNetwork:
        """
        Build an untrained irregular summary network.

        Parameters
        ----------
        shape : IrregularSummaryShape
            The input and output dimensions.

        Returns
        -------
        IrregularSummaryNetwork
            The network.
        """
        return self.head.build_summary_network(self.build_encoder(shape.input_channels), shape)

    def build_irregular_interpolation_network(self, shape: IrregularSequenceShape) -> IrregularInterpolationNetwork:
        """
        Build an untrained irregular interpolation network.

        Parameters
        ----------
        shape : IrregularSequenceShape
            The dimensions of the sequences.

        Returns
        -------
        IrregularInterpolationNetwork
            The network.
        """
        return MTANInterpolationNetwork(
            channels=shape.channels,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            projection_size=self.projection_size,
            time_embedding_size=self.time_embedding_size,
            num_heads=self.num_heads,
            num_reference_points=self.num_reference_points,
            observation_std=self.observation_std,
            kl_weight=self.kl_weight,
        )
