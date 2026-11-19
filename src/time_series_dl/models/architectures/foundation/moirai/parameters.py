from dataclasses import dataclass, replace
from typing import Self

from ....components.pretrained import PretrainedWeights, PretrainedWeightsLoader
from ....interfaces import ForecastNetwork, ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .forecast_network import MoiraiForecastNetwork
from .official_names import MoiraiOfficialNames
from .size import MoiraiSize

_HEAD_SIZE: int = 64


@dataclass(frozen=True, kw_only=True)
class MoiraiParameters:
    """
    Moirai (Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", ICML 2024), release
    1.1-R: a pretrained masked Transformer encoder over patches of any number of variates, with any-variate attention,
    several patch sizes and a mixture distribution output, forecasting zero-shot by sampling.

    The published weights (``Salesforce/moirai-1.1-R-*``) are licensed CC-BY-NC-4.0 and may be used for
    non-commercial purposes only; this code does not change that license.

    The defaults are the architecture of ``Salesforce/moirai-1.1-R-small`` without its weights; ``pretrained`` gives
    the architecture and weights of each published checkpoint. As in the original, there are ``model_size // 64``
    attention heads and a gated feed-forward network of about ``8/3 * model_size`` hidden units.

    Attributes
    ----------
    model_size : int
        Size of the hidden states (``d_model``); a multiple of 64.
    num_layers : int
        Number of encoder layers.
    patch_sizes : tuple[int, ...]
        Patch sizes with their own input and output projections, strictly increasing.
    patch_size : int | None
        Patch size of every forecast; None to select one per window as the original's ``patch_size="auto"``, which
        needs ``input_length > horizon``. The paper recommends 8 for yearly and quarterly, 8 to 32 for monthly, 16
        or 32 for weekly and daily, 32 or 64 for hourly and 32 to 128 for minute-level data.
    is_scaled : bool
        Whether each variate is standardized by the mean and standard deviation of its past values.
    normal_scale : float
        Fixed standard deviation of the normal component of the mixture.
    attention_dropout : float
        Dropout probability of the attention weights, active only while fine-tuning.
    dropout : float
        Dropout probability of the residual branches, active only while fine-tuning.
    layer_norm_epsilon : float
        Epsilon of the root-mean-square normalizations.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If model_size is not a positive multiple of 64, a count or the normal scale is not positive, the patch sizes
        are invalid, patch_size is not one of them, or a dropout is not in [0, 1).
    """

    model_size: int = 384
    num_layers: int = 6
    patch_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    patch_size: int | None = None
    is_scaled: bool = True
    normal_scale: float = 1e-3
    attention_dropout: float = 0.0
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        if self.model_size <= 0 or self.model_size % _HEAD_SIZE != 0:
            raise ValueError(f"model_size must be a positive multiple of {_HEAD_SIZE}, got {self.model_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not self.patch_sizes or any(size <= 0 for size in self.patch_sizes):
            raise ValueError(f"patch_sizes must be positive and not empty, got {self.patch_sizes}")
        if any(lower >= upper for lower, upper in zip(self.patch_sizes, self.patch_sizes[1:], strict=False)):
            raise ValueError(f"patch_sizes must be strictly increasing, got {self.patch_sizes}")
        if self.patch_size is not None and self.patch_size not in self.patch_sizes:
            raise ValueError(f"patch_size must be one of {self.patch_sizes}, got {self.patch_size}")
        if self.normal_scale <= 0.0:
            raise ValueError(f"normal_scale must be positive, got {self.normal_scale}")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError(f"layer_norm_epsilon must be positive, got {self.layer_norm_epsilon}")
        for name, value in (("attention_dropout", self.attention_dropout), ("dropout", self.dropout)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

    @classmethod
    def pretrained(cls, size: MoiraiSize, patch_size: int | None = None) -> Self:
        """
        Get the architecture and the published weights (CC-BY-NC-4.0, non-commercial) of a Moirai 1.1-R
        checkpoint.

        Parameters
        ----------
        size : MoiraiSize
            The checkpoint.
        patch_size : int | None
            Patch size of every forecast; None to select one per window.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=size.repository, revision=size.revision)
        match size:
            case MoiraiSize.SMALL:
                return cls(model_size=384, num_layers=6, patch_size=patch_size, weights=weights)
            case MoiraiSize.BASE:
                return cls(model_size=768, num_layers=12, patch_size=patch_size, weights=weights)
            case MoiraiSize.LARGE:
                return cls(model_size=1024, num_layers=24, patch_size=patch_size, weights=weights)

    @property
    def num_heads(self) -> int:
        """
        Get the number of attention heads.

        Returns
        -------
        int
            ``model_size // 64``.
        """
        return self.model_size // _HEAD_SIZE

    @property
    def feedforward_size(self) -> int:
        """
        Get the hidden size of the gated feed-forward networks.

        Returns
        -------
        int
            Two thirds of ``4 * model_size``, rounded up to a multiple of 8.
        """
        return (int(4 * self.model_size * 2 / 3) + 7) // 8 * 8

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
            None: the network outputs sample paths, read with ``SampleObjective``.
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
        Build a generative forecast network, loading the published weights if ``weights`` is set.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward``.

        Returns
        -------
        GenerativeForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If the patch size is selected automatically and ``input_length`` does not exceed ``horizon``.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: MoiraiForecastNetwork = MoiraiForecastNetwork(self, shape)
        if self.weights is not None:
            PretrainedWeightsLoader.load(network, self.weights, MoiraiOfficialNames.mapping())
        return network

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build the generative forecast network of ``build_generative_forecast_network``.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward``.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return self.build_generative_forecast_network(shape)
