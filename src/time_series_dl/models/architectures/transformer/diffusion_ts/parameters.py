from dataclasses import dataclass, field

from ....interfaces import SequenceShape
from ....interfaces.sequence_generation_network import SequenceGenerationNetwork
from .cosine_noise_schedule import CosineNoiseSchedule
from .loss_kind import DiffusionTSLossKind
from .sequence_generation_network import DiffusionTSSequenceGenerationNetwork


@dataclass(frozen=True, kw_only=True)
class DiffusionTSParameters:
    """
    Diffusion-TS (Yuan and Qiao, 2024): a denoising diffusion model whose encoder-decoder Transformer estimates the
    clean sequence as an interpretable sum of a polynomial trend, Fourier seasonality and a residual, trained with a
    time and frequency domain loss.

    The defaults are those of the authors' configuration for stock data: 2 encoder and 2 decoder layers of size 64
    with 4 heads, a feed-forward network 4 times wider, no dropout, a cubic trend, 500 diffusion steps of the cosine
    schedule, the absolute error, and a Fourier weight of ``sqrt(length) / 5``.

    Attributes
    ----------
    model_size : int
        Size of each token vector, also of the diffusion step embedding; even and at least 4.
    num_heads : int
        Number of attention heads; it divides model_size.
    encoder_layers : int
        Number of encoder layers.
    decoder_layers : int
        Number of decoder layers.
    feedforward_factor : int
        Hidden size of the feed-forward networks as a multiple of model_size.
    dropout : float
        Dropout probability of the attention weights, residual branches and embeddings.
    trend_degree : int
        Highest power of the polynomial trend of each decoder layer.
    seasonality_factor : float
        Each decoder layer keeps ``seasonality_factor * log(number of frequencies)`` Fourier components (at least
        one).
    noise_schedule : CosineNoiseSchedule
        Variances of the diffusion process.
    loss_kind : DiffusionTSLossKind
        Error of each value in the time and frequency domain.
    fourier_weight : float | None
        Weight of the frequency domain error; ``sqrt(length) / 5`` when None.

    Raises
    ------
    ValueError
        If a size or count is not positive, model_size is odd or below 4 or not divisible by num_heads, dropout is not
        in [0, 1), or a weight or factor is negative.
    """

    model_size: int = 64
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_factor: int = 4
    dropout: float = 0.0
    trend_degree: int = 3
    seasonality_factor: float = 1.0
    noise_schedule: CosineNoiseSchedule = field(default_factory=CosineNoiseSchedule)
    loss_kind: DiffusionTSLossKind = DiffusionTSLossKind.ABSOLUTE
    fourier_weight: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("num_heads", self.num_heads),
            ("encoder_layers", self.encoder_layers),
            ("decoder_layers", self.decoder_layers),
            ("feedforward_factor", self.feedforward_factor),
            ("trend_degree", self.trend_degree),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size < 4 or self.model_size % 2 != 0:
            raise ValueError(f"model_size must be even and at least 4, got {self.model_size}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads must divide model_size, got {self.num_heads} and {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.seasonality_factor < 0.0:
            raise ValueError(f"seasonality_factor must not be negative, got {self.seasonality_factor}")
        if self.fourier_weight is not None and self.fourier_weight < 0.0:
            raise ValueError(f"fourier_weight must not be negative when set, got {self.fourier_weight}")

    def build_sequence_generation_network(self, shape: SequenceShape) -> SequenceGenerationNetwork:
        """
        Build an untrained Diffusion-TS network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences it generates.

        Returns
        -------
        SequenceGenerationNetwork
            The network.
        """
        return DiffusionTSSequenceGenerationNetwork(self, shape)
