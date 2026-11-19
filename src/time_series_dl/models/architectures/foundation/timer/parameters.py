from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TimerForecastNetwork
from .official_names import TimerOfficialNames
from .size import TimerSize


@dataclass(frozen=True, kw_only=True)
class TimerParameters:
    """
    Timer (Liu et al., "Timer: Generative Pre-trained Transformers Are Large Time Series Models", ICML 2024): a
    pretrained decoder-only Transformer over patches that forecasts the next patch of a normalized series and
    generates longer horizons patch by patch.

    The defaults are the architecture of ``thuml/timer-base-84m`` without its weights; ``pretrained`` gives the
    architecture and weights of the published checkpoint.

    Attributes
    ----------
    patch_length : int
        Number of values per input patch (``input_token_len``).
    output_patch_length : int
        Number of values each position predicts (``output_token_lens``); a multiple of patch_length.
    model_size : int
        Size of the hidden states.
    feedforward_size : int
        Hidden size of the gated feed-forward networks.
    num_layers : int
        Number of decoder layers.
    num_heads : int
        Number of attention heads; must divide model_size into an even head size.
    rotary_base : float
        Base of the rotary position embedding frequencies.
    attention_dropout : float
        Dropout probability of the attention weights, active only while fine-tuning.
    layer_norm_epsilon : float
        Epsilon of the layer normalizations.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size into an even head size, output_patch_length
        is not a multiple of patch_length, or the dropout or epsilon are out of range.
    """

    patch_length: int = 96
    output_patch_length: int = 96
    model_size: int = 1024
    feedforward_size: int = 2048
    num_layers: int = 8
    num_heads: int = 8
    rotary_base: float = 10000.0
    attention_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("patch_length", self.patch_length),
            ("output_patch_length", self.output_patch_length),
            ("model_size", self.model_size),
            ("feedforward_size", self.feedforward_size),
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0 or (self.model_size // self.num_heads) % 2 != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size} into an even size")
        if self.output_patch_length % self.patch_length != 0:
            raise ValueError(
                f"output_patch_length {self.output_patch_length} must be a multiple of patch_length {self.patch_length}"
            )
        if self.rotary_base <= 1.0:
            raise ValueError(f"rotary_base must exceed 1, got {self.rotary_base}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError(f"layer_norm_epsilon must be positive, got {self.layer_norm_epsilon}")

    @classmethod
    def pretrained(cls, size: TimerSize) -> Self:
        """
        Get the architecture and the published weights of a Timer checkpoint.

        Parameters
        ----------
        size : TimerSize
            The checkpoint.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=size.repository, revision=size.revision)
        match size:
            case TimerSize.BASE:
                return cls(weights=weights)

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
            An empty tuple: Timer forecasts points only.
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

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build a forecast network, loading the published weights if ``weights`` is set.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` must be 1 and ``input_length`` at least one patch.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If the shape does not fit Timer.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: TimerForecastNetwork = TimerForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, TimerOfficialNames.mapping())
        return network
