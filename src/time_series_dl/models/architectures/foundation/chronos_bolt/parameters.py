from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....components.t5 import T5Configuration, T5FeedForwardKind
from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import ChronosBoltForecastNetwork
from .official_names import ChronosBoltOfficialNames
from .size import ChronosBoltSize


@dataclass(frozen=True, kw_only=True)
class ChronosBoltParameters:
    """
    Chronos-Bolt (Ansari et al., "Chronos: Learning the Language of Time Series", TMLR 2024; Chronos-Bolt release,
    2024): a pretrained T5 encoder-decoder over patches of standardized series that forecasts several quantiles of a
    whole block of future values in one step.

    The defaults are the architecture of ``amazon/chronos-bolt-small`` without its weights; ``pretrained`` gives the
    architecture and weights of each published checkpoint.

    Attributes
    ----------
    model_size : int
        Size of the hidden states (``d_model``).
    key_value_size : int
        Size of each attention head (``d_kv``).
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks and the residual blocks (``d_ff``).
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
    context_length : int
        Maximum number of past values read; longer windows are cut to their last ``context_length`` values.
    patch_length : int
        Number of values per input patch.
    patch_stride : int
        Step between input patches.
    prediction_length : int
        Number of future values forecast in one step; longer horizons are forecast in blocks.
    quantile_levels : tuple[float, ...]
        Levels of the forecast quantiles, strictly increasing in (0, 1).
    is_register_token_used : bool
        Whether a register token is appended to the encoder inputs.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, the patch stride exceeds the patch length, the quantile levels are invalid, or
        dropout is not in [0, 1).
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
    context_length: int = 2048
    patch_length: int = 16
    patch_stride: int = 16
    prediction_length: int = 64
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    is_register_token_used: bool = True
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("context_length", self.context_length),
            ("patch_length", self.patch_length),
            ("patch_stride", self.patch_stride),
            ("prediction_length", self.prediction_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.patch_stride > self.patch_length:
            raise ValueError(f"patch_stride {self.patch_stride} must not exceed patch_length {self.patch_length}")
        if not self.quantile_levels:
            raise ValueError("quantile_levels must not be empty.")
        if any(not 0.0 < level < 1.0 for level in self.quantile_levels):
            raise ValueError(f"quantile_levels must be in (0, 1), got {self.quantile_levels}")
        if any(lower >= upper for lower, upper in zip(self.quantile_levels, self.quantile_levels[1:], strict=False)):
            raise ValueError(f"quantile_levels must be strictly increasing, got {self.quantile_levels}")
        # The stack configurations validate the remaining sizes and the dropout.
        _ = self.encoder_configuration
        _ = self.decoder_configuration

    @classmethod
    def pretrained(cls, size: ChronosBoltSize) -> Self:
        """
        Get the architecture and the published weights of a Chronos-Bolt checkpoint.

        Parameters
        ----------
        size : ChronosBoltSize
            The checkpoint.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=size.repository, revision=size.revision)
        match size:
            case ChronosBoltSize.TINY:
                return cls(
                    model_size=256,
                    num_heads=4,
                    feedforward_size=1024,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    weights=weights,
                )
            case ChronosBoltSize.MINI:
                return cls(
                    model_size=384,
                    num_heads=8,
                    feedforward_size=1536,
                    num_encoder_layers=4,
                    num_decoder_layers=4,
                    weights=weights,
                )
            case ChronosBoltSize.SMALL:
                return cls(weights=weights)
            case ChronosBoltSize.BASE:
                return cls(
                    model_size=768,
                    num_heads=12,
                    feedforward_size=3072,
                    num_encoder_layers=12,
                    num_decoder_layers=12,
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
            ``quantile_levels``.
        """
        return self.quantile_levels

    def without_pretrained_weights(self) -> Self:
        """
        Get the same architecture without published weights.

        Returns
        -------
        Self
            The parameters with ``weights`` set to None.
        """
        return replace(self, weights=None)

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build a forecast network, loading the published weights if ``weights`` is set.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` must be 1 (median) or the number of quantile levels.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If ``num_parameters`` does not fit the quantile levels.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: ChronosBoltForecastNetwork = ChronosBoltForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, ChronosBoltOfficialNames.mapping())
        return network

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
