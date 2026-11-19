from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TTMForecastNetwork
from .mixer_configuration import TTMMixerConfiguration
from .official_names import TTMOfficialNames
from .variant import TTMVariant


@dataclass(frozen=True, kw_only=True)
class TTMParameters:
    """
    TinyTimeMixer (Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced
    Zero/Few-Shot Forecasting of Multivariate Time Series", NeurIPS 2024): a pretrained, channel-independent TSMixer
    with adaptive patching and a small mixer decoder that forecasts a fixed number of future values in one step.

    The defaults are the architecture of the ``main`` branch of ``ibm-granite/granite-timeseries-ttm-r2`` (context
    512, prediction 96) without its weights; ``pretrained`` gives the architecture and weights of each published r2
    variant.

    Attributes
    ----------
    context_length : int
        Number of past values read; longer windows are cut and shorter ones padded with zeros on the left.
    prediction_length : int
        Number of future values forecast; the horizon must not exceed it.
    patch_length : int
        Number of values per patch.
    patch_stride : int
        Step between patches.
    model_size : int
        Number of features per patch in the backbone (``d_model``).
    num_layers : int
        Number of backbone mixer layers per adaptive patching level.
    adaptive_patching_levels : int
        Number of adaptive patching levels of the backbone; zero for none.
    expansion_factor : int
        Ratio of the perceptrons' hidden sizes to their input sizes.
    dropout : float
        Dropout probability of the backbone, active only while fine-tuning.
    decoder_model_size : int
        Number of features per patch in the decoder.
    decoder_num_layers : int
        Number of decoder mixer layers per adaptive patching level.
    decoder_adaptive_patching_levels : int
        Number of adaptive patching levels of the decoder; zero for none.
    head_dropout : float
        Dropout probability of the decoder and the head, active only while fine-tuning.
    is_gated_attention_used : bool
        Whether a gated attention follows each perceptron.
    layer_norm_epsilon : float
        Epsilon of the layer normalizations.
    scale_epsilon : float
        Added to the variance of each context before its standard deviation is taken.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, the context is not longer than one patch, the stride exceeds the patch length, an
        adaptive patching level does not divide its model size, or a dropout or epsilon is out of range.
    """

    context_length: int = 512
    prediction_length: int = 96
    patch_length: int = 64
    patch_stride: int = 64
    model_size: int = 192
    num_layers: int = 2
    adaptive_patching_levels: int = 3
    expansion_factor: int = 2
    dropout: float = 0.4
    decoder_model_size: int = 128
    decoder_num_layers: int = 2
    decoder_adaptive_patching_levels: int = 0
    head_dropout: float = 0.4
    is_gated_attention_used: bool = True
    layer_norm_epsilon: float = 1e-5
    scale_epsilon: float = 1e-5
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("context_length", self.context_length),
            ("prediction_length", self.prediction_length),
            ("patch_length", self.patch_length),
            ("patch_stride", self.patch_stride),
            ("model_size", self.model_size),
            ("num_layers", self.num_layers),
            ("expansion_factor", self.expansion_factor),
            ("decoder_model_size", self.decoder_model_size),
            ("decoder_num_layers", self.decoder_num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.context_length <= self.patch_length:
            raise ValueError(f"context_length {self.context_length} must exceed patch_length {self.patch_length}")
        if self.patch_stride > self.patch_length:
            raise ValueError(f"patch_stride {self.patch_stride} must not exceed patch_length {self.patch_length}")
        if self.adaptive_patching_levels < 0 or self.decoder_adaptive_patching_levels < 0:
            raise ValueError("adaptive patching levels must not be negative.")
        for configuration in (self.encoder_configuration, self.decoder_configuration):
            for level in range(configuration.adaptive_patching_levels):
                if configuration.size % configuration.compute_patch_factor(level) != 0:
                    raise ValueError(
                        f"model size {configuration.size} must be divisible by 2 ** {level} for adaptive patching"
                    )
        for name, probability in (("dropout", self.dropout), ("head_dropout", self.head_dropout)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {probability}")
        if self.layer_norm_epsilon <= 0.0 or self.scale_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon and scale_epsilon must be positive.")

    @classmethod
    def pretrained(cls, variant: TTMVariant) -> Self:
        """
        Get the architecture and the published weights of a TinyTimeMixer r2 variant.

        Parameters
        ----------
        variant : TTMVariant
            The checkpoint.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=variant.repository, revision=variant.revision)
        is_short: bool = variant.context_length == 512
        return cls(
            context_length=variant.context_length,
            prediction_length=variant.prediction_length,
            patch_length=64 if is_short else 128,
            patch_stride=64 if is_short else 128,
            model_size=192 if is_short else 384,
            decoder_model_size=128 if is_short else 256,
            weights=weights,
        )

    @property
    def num_patches(self) -> int:
        """
        Get the number of patches of the context.

        Returns
        -------
        int
            The number of whole patches that fit, ending at the last value.
        """
        return (self.context_length - self.patch_length) // self.patch_stride + 1

    @property
    def encoder_configuration(self) -> TTMMixerConfiguration:
        """
        Get the sizes of the backbone.

        Returns
        -------
        TTMMixerConfiguration
            The configuration.
        """
        return TTMMixerConfiguration(
            num_patches=self.num_patches,
            size=self.model_size,
            num_layers=self.num_layers,
            adaptive_patching_levels=self.adaptive_patching_levels,
            expansion_factor=self.expansion_factor,
            dropout=self.dropout,
            is_gated=self.is_gated_attention_used,
            epsilon=self.layer_norm_epsilon,
        )

    @property
    def decoder_configuration(self) -> TTMMixerConfiguration:
        """
        Get the sizes of the decoder.

        Returns
        -------
        TTMMixerConfiguration
            The configuration.
        """
        return TTMMixerConfiguration(
            num_patches=self.num_patches,
            size=self.decoder_model_size,
            num_layers=self.decoder_num_layers,
            adaptive_patching_levels=self.decoder_adaptive_patching_levels,
            expansion_factor=self.expansion_factor,
            dropout=self.head_dropout,
            is_gated=self.is_gated_attention_used,
            epsilon=self.layer_norm_epsilon,
        )

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
            An empty tuple: TinyTimeMixer forecasts points only.
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
            The input and output dimensions; ``num_parameters`` must be 1 and the horizon at most
            ``prediction_length``.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If the shape does not fit the model.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: TTMForecastNetwork = TTMForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, TTMOfficialNames.mapping(self))
        return network
