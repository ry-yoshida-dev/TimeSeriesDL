from dataclasses import dataclass, field

from ....components.diffusion import NoiseSchedule, NoiseScheduleKind
from ....interfaces import ForecastNetwork, ForecastShape, SequenceShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .conditional_imputer import CSDIConditionalImputer
from .generative_forecast_network import CSDIForecastNetwork
from .masked_reconstruction_network import CSDIMaskedReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class CSDIParameters:
    """
    CSDI (Tashiro et al., 2021): a conditional score-based diffusion model that imputes the unobserved values of a
    (features, time) array from the observed ones, with a Transformer over time and one over features in every
    residual layer. Forecasting imputes the horizon of every target channel from the past window; imputation trains
    with random conditional masks and fills each missing value with the median of samples.

    The defaults are those of the authors' forecasting configuration: 4 residual layers of 64 channels, 8 attention
    heads with a feed-forward size of 64, 128 sinusoidal diffusion step features, 128 time embedding and 16 feature
    embedding dimensions, and 50 diffusion steps with quadratically growing variances from 1e-4 to 0.5 (the same in
    the authors' imputation configuration). The paper imputes with the median of 100 samples; the default of 20
    makes imputation five times faster (each sample runs every diffusion step), at the cost of a slightly noisier
    median.

    Attributes
    ----------
    channels : int
        Number of channels of the residual layers, also the Transformers' model size.
    num_layers : int
        Number of residual layers.
    num_heads : int
        Number of attention heads of each Transformer.
    feedforward_size : int
        Hidden size of the Transformers' feed-forward layers.
    dropout : float
        Dropout probability of the Transformers.
    step_feature_size : int
        Number of sinusoidal features of the diffusion step, also the size of its embedding; even and at least 4.
    time_embedding_size : int
        Size of the sinusoidal embedding of each time position; even.
    feature_embedding_size : int
        Size of the learned embedding of each feature (channel).
    noise_schedule : NoiseSchedule
        Variances of the diffusion process.
    num_imputation_samples : int
        Number of samples whose median imputes each missing value outside training.

    Raises
    ------
    ValueError
        If a size or num_imputation_samples is not positive, channels is not divisible by num_heads, dropout is not in
        [0, 1), or an embedding size that must be even is odd.
    """

    channels: int = 64
    num_layers: int = 4
    num_heads: int = 8
    feedforward_size: int = 64
    dropout: float = 0.1
    step_feature_size: int = 128
    time_embedding_size: int = 128
    feature_embedding_size: int = 16
    noise_schedule: NoiseSchedule = field(
        default_factory=lambda: NoiseSchedule(
            kind=NoiseScheduleKind.QUADRATIC, num_steps=50, first_variance=1e-4, last_variance=0.5
        )
    )
    num_imputation_samples: int = 20

    def __post_init__(self) -> None:
        for name, value in (
            ("channels", self.channels),
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("feedforward_size", self.feedforward_size),
            ("time_embedding_size", self.time_embedding_size),
            ("feature_embedding_size", self.feature_embedding_size),
            ("num_imputation_samples", self.num_imputation_samples),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.channels % self.num_heads != 0:
            raise ValueError(f"channels ({self.channels}) must be divisible by num_heads ({self.num_heads})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.step_feature_size < 4 or self.step_feature_size % 2 != 0:
            raise ValueError(f"step_feature_size must be even and at least 4, got {self.step_feature_size}")
        if self.time_embedding_size % 2 != 0:
            raise ValueError(f"time_embedding_size must be even, got {self.time_embedding_size}")

    def build_conditional_imputer(self, num_features: int) -> CSDIConditionalImputer:
        """
        Build an untrained mask-aware imputer of (features, time) arrays, for forecasting and imputation alike.

        Parameters
        ----------
        num_features : int
            Number of features (channels) of the arrays.

        Returns
        -------
        CSDIConditionalImputer
            The imputer.
        """
        return CSDIConditionalImputer(self, num_features)

    def build_masked_reconstruction_network(self, shape: SequenceShape) -> MaskedReconstructionNetwork:
        """
        Build an untrained CSDI imputation network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        MaskedReconstructionNetwork
            The network.
        """
        return CSDIMaskedReconstructionNetwork(self, shape)

    def build_generative_forecast_network(self, shape: ForecastShape) -> GenerativeForecastNetwork:
        """
        Build an untrained CSDI forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward``.

        Returns
        -------
        GenerativeForecastNetwork
            The network.
        """
        return CSDIForecastNetwork(self, shape)

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained CSDI forecast network, whose outputs are sample paths.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return self.build_generative_forecast_network(shape)
