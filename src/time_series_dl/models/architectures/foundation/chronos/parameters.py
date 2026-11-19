from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....components.t5 import T5Configuration, T5FeedForwardKind
from ....interfaces import ForecastNetwork, ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .forecast_network import ChronosForecastNetwork
from .official_names import ChronosOfficialNames
from .size import ChronosSize


@dataclass(frozen=True, kw_only=True)
class ChronosParameters:
    """
    Chronos (Ansari et al., "Chronos: Learning the Language of Time Series", TMLR 2024): a pretrained T5
    encoder-decoder language model over the tokens of mean-scaled, uniformly quantized series, which forecasts by
    sampling the next tokens.

    The defaults are the architecture and generation settings of ``amazon/chronos-t5-small`` without its weights;
    ``pretrained`` gives the architecture and weights of each published checkpoint.

    Attributes
    ----------
    model_size : int
        Size of the hidden states (``d_model``).
    key_value_size : int
        Size of each attention head (``d_kv``).
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks (``d_ff``).
    num_encoder_layers : int
        Number of T5 encoder blocks.
    num_decoder_layers : int
        Number of T5 decoder blocks.
    relative_attention_num_buckets : int
        Number of buckets of relative positions.
    relative_attention_max_distance : int
        Distance beyond which relative positions share a bucket.
    dropout : float
        Dropout probability, active only while fine-tuning.
    layer_norm_epsilon : float
        Epsilon of the layer normalizations.
    num_tokens : int
        Size of the vocabulary: the PAD and EOS tokens and ``num_tokens - 2`` value tokens (``n_tokens``).
    low_limit : float
        Lowest bin center, in units of the mean absolute value of the context.
    high_limit : float
        Highest bin center, in units of the mean absolute value of the context.
    context_length : int
        Maximum number of past values read; longer windows are cut to their last ``context_length`` values.
    prediction_length : int
        Number of future values sampled per block; longer horizons are sampled in blocks, each continuing the
        context with the median of the previous block.
    temperature : float
        Temperature of the token distribution while sampling.
    top_k : int
        Number of most likely tokens sampling draws from.
    top_p : float
        Probability mass of the most likely tokens sampling draws from (nucleus sampling); 1 for no limit.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, the vocabulary has no room for value tokens, the limits are not increasing, the
        temperature is not positive, top_p is not in (0, 1], or dropout is not in [0, 1).
    """

    model_size: int = 512
    key_value_size: int = 64
    num_heads: int = 8
    feedforward_size: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-6
    num_tokens: int = 4096
    low_limit: float = -15.0
    high_limit: float = 15.0
    context_length: int = 512
    prediction_length: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("context_length", self.context_length),
            ("prediction_length", self.prediction_length),
            ("top_k", self.top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_tokens < 5:
            raise ValueError(f"num_tokens must leave at least two value tokens, got {self.num_tokens}")
        if self.low_limit >= self.high_limit:
            raise ValueError(f"low_limit {self.low_limit} must be below high_limit {self.high_limit}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        # The stack configurations validate the remaining sizes and the dropout.
        _ = self.encoder_configuration
        _ = self.decoder_configuration

    @classmethod
    def pretrained(cls, size: ChronosSize) -> Self:
        """
        Get the architecture and the published weights of a Chronos checkpoint.

        Parameters
        ----------
        size : ChronosSize
            The checkpoint.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=size.repository, revision=size.revision)
        match size:
            case ChronosSize.TINY:
                return cls(
                    model_size=256,
                    num_heads=4,
                    feedforward_size=1024,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    weights=weights,
                )
            case ChronosSize.MINI:
                return cls(
                    model_size=384,
                    num_heads=8,
                    feedforward_size=1536,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    weights=weights,
                )
            case ChronosSize.SMALL:
                return cls(weights=weights)
            case ChronosSize.BASE:
                return cls(
                    model_size=768,
                    num_heads=12,
                    feedforward_size=3072,
                    num_encoder_layers=12,
                    num_decoder_layers=12,
                    weights=weights,
                )
            case ChronosSize.LARGE:
                return cls(
                    model_size=1024,
                    num_heads=16,
                    feedforward_size=4096,
                    num_encoder_layers=24,
                    num_decoder_layers=24,
                    weights=weights,
                )

    @property
    def encoder_configuration(self) -> T5Configuration:
        """
        Get the sizes of the T5 encoder.

        Returns
        -------
        T5Configuration
            The configuration.
        """
        return self._stack_configuration(self.num_encoder_layers)

    @property
    def decoder_configuration(self) -> T5Configuration:
        """
        Get the sizes of the T5 decoder.

        Returns
        -------
        T5Configuration
            The configuration.
        """
        return self._stack_configuration(self.num_decoder_layers)

    @property
    def is_pretrained(self) -> bool:
        """
        Check whether the builders load published weights.

        Returns
        -------
        bool
            True if ``weights`` is set.
        """
        return self.weights is not None

    @property
    def zero_shot_quantile_levels(self) -> tuple[float, ...]:
        """
        Get the levels of the quantiles the forecast network outputs.

        Returns
        -------
        tuple[float, ...]
            Empty: the network outputs sample paths, which ``SampleObjective`` summarizes.
        """
        return ()

    def without_pretrained_weights(self) -> Self:
        """
        Get the same architecture without published weights.

        Returns
        -------
        Self
            The parameters with ``weights`` set to None.
        """
        return replace(self, weights=None)

    def build_generative_forecast_network(self, shape: ForecastShape) -> GenerativeForecastNetwork:
        """
        Build a forecast network, loading the published weights if ``weights`` is set.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward`` (the
            official pipeline draws 20).

        Returns
        -------
        GenerativeForecastNetwork
            The network.

        Raises
        ------
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: ChronosForecastNetwork = ChronosForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, ChronosOfficialNames.mapping())
        return network

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build the generative forecast network, whose outputs are sample paths.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths.

        Returns
        -------
        ForecastNetwork
            The network of ``build_generative_forecast_network``.
        """
        return self.build_generative_forecast_network(shape)

    def _stack_configuration(self, num_layers: int) -> T5Configuration:
        return T5Configuration(
            model_size=self.model_size,
            key_value_size=self.key_value_size,
            num_heads=self.num_heads,
            feedforward_size=self.feedforward_size,
            num_layers=num_layers,
            feed_forward_kind=T5FeedForwardKind.RELU,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
