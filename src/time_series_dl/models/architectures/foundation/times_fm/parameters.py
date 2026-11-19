from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TimesFMForecastNetwork
from .frequency import TimesFMFrequency
from .official_names import TimesFMOfficialNames
from .release import TimesFMRelease


@dataclass(frozen=True, kw_only=True)
class TimesFMParameters:
    """
    TimesFM (Das et al., "A decoder-only foundation model for time-series forecasting", ICML 2024), version 2.0: a
    pretrained decoder-only transformer over patches that forecasts the mean and nine quantiles of the next block of
    values.

    The defaults are the architecture of ``google/timesfm-2.0-500m-pytorch`` without its weights; ``pretrained``
    gives the architecture and weights of a published checkpoint.

    Attributes
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of attention heads.
    head_size : int
        Size of each attention head.
    feedforward_size : int
        Hidden size of the feed-forward networks and the residual blocks.
    num_layers : int
        Number of decoder layers.
    layer_norm_epsilon : float
        Epsilon of the root-mean-square normalization before attention.
    context_length : int
        Maximum number of past values read; longer windows are cut to their last ``context_length`` values.
    patch_length : int
        Number of values per input patch.
    output_patch_length : int
        Number of future values forecast in one step; longer horizons are decoded autoregressively.
    quantile_levels : tuple[float, ...]
        Levels of the quantile heads, strictly increasing in (0, 1).
    num_frequencies : int
        Number of frequency categories with an embedding.
    frequency : TimesFMFrequency
        Frequency category of the forecast series.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, the context is shorter than one patch, the quantile levels are invalid, or the
        frequency has no embedding.
    """

    model_size: int = 1280
    num_heads: int = 16
    head_size: int = 80
    feedforward_size: int = 1280
    num_layers: int = 50
    layer_norm_epsilon: float = 1e-6
    context_length: int = 2048
    patch_length: int = 32
    output_patch_length: int = 128
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    num_frequencies: int = 3
    frequency: TimesFMFrequency = TimesFMFrequency.HIGH
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("head_size", self.head_size),
            ("feedforward_size", self.feedforward_size),
            ("num_layers", self.num_layers),
            ("context_length", self.context_length),
            ("patch_length", self.patch_length),
            ("output_patch_length", self.output_patch_length),
            ("num_frequencies", self.num_frequencies),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError(f"layer_norm_epsilon must be positive, got {self.layer_norm_epsilon}")
        if self.context_length < self.patch_length:
            raise ValueError(f"context_length {self.context_length} must be at least patch_length {self.patch_length}")
        if not self.quantile_levels:
            raise ValueError("quantile_levels must not be empty.")
        if any(not 0.0 < level < 1.0 for level in self.quantile_levels):
            raise ValueError(f"quantile_levels must be in (0, 1), got {self.quantile_levels}")
        if any(lower >= upper for lower, upper in zip(self.quantile_levels, self.quantile_levels[1:], strict=False)):
            raise ValueError(f"quantile_levels must be strictly increasing, got {self.quantile_levels}")
        if self.frequency.index >= self.num_frequencies:
            raise ValueError(f"frequency {self.frequency} needs more than {self.num_frequencies} frequencies")

    @classmethod
    def pretrained(cls, release: TimesFMRelease, frequency: TimesFMFrequency = TimesFMFrequency.HIGH) -> Self:
        """
        Get the architecture and the published weights of a TimesFM checkpoint.

        Parameters
        ----------
        release : TimesFMRelease
            The checkpoint.
        frequency : TimesFMFrequency
            Frequency category of the series to forecast.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=release.repository, revision=release.revision)
        match release:
            case TimesFMRelease.V2_0_500M:
                return cls(frequency=frequency, weights=weights)

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
            The input and output dimensions; ``num_parameters`` must be 1 (mean) or the number of quantile levels.

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
        network: TimesFMForecastNetwork = TimesFMForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, TimesFMOfficialNames.mapping())
        return network
