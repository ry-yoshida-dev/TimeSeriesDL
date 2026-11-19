from dataclasses import dataclass, replace
from typing import Self

from torch import nn

from ....components.pretrained import ParameterNameMapping, PretrainedWeights, PretrainedWeightsLoader
from ....components.t5 import T5Configuration, T5FeedForwardKind
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    MaskedReconstructionNetwork,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import MOMENTForecastNetwork
from .masked_reconstruction_network import MOMENTMaskedReconstructionNetwork
from .official_names import MOMENTOfficialNames
from .reconstruction_network import MOMENTReconstructionNetwork
from .size import MOMENTSize
from .summary_network import MOMENTSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class MOMENTParameters:
    """
    MOMENT (Goswami et al., "MOMENT: A Family of Open Time-series Foundation Models", ICML 2024): a T5 encoder over
    patches of instance-normalized univariate series of a fixed length, pretrained to reconstruct masked patches.

    The defaults are the architecture of ``AutonLab/MOMENT-1-large`` without its weights; ``pretrained`` gives the
    architecture and weights of each published checkpoint. Only the reconstruction head is pretrained, so
    reconstruction networks work zero-shot, while forecast and summary networks get a new head to fine-tune.

    Attributes
    ----------
    model_size : int
        Size of the patch embeddings and encoder states (``d_model``).
    key_value_size : int
        Size of each attention head (``d_kv``).
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the gated-GELU feed-forward networks (``d_ff``).
    num_layers : int
        Number of T5 encoder blocks.
    relative_attention_num_buckets : int
        Number of buckets of relative positions.
    relative_attention_max_distance : int
        Distance beyond which relative positions share a bucket.
    dropout : float
        Dropout probability of the T5 encoder.
    layer_norm_epsilon : float
        Epsilon of the encoder's layer normalizations.
    input_length : int
        Fixed number of time steps the encoder reads (``seq_len``).
    patch_length : int
        Number of values per non-overlapping patch; must divide input_length.
    patch_dropout : float
        Dropout probability of the patch embeddings.
    head_dropout : float
        Dropout probability of the heads.
    normalization_epsilon : float
        Added to the standard deviation of the instance normalization.
    is_positional_embedding_used : bool
        Whether the sinusoidal position embedding is added to the patch embeddings.
    is_backbone_frozen : bool
        Whether the patch embedding and the encoder are frozen, so that fine-tuning trains only the head, as the
        official fine-tuning of the published checkpoints does by default; ``pretrained`` sets it.
    weights : PretrainedWeights | None
        Published weights loaded into every built network; None for random initialization.

    Raises
    ------
    ValueError
        If a size is not positive, patch_length does not divide input_length, or a dropout or epsilon is out of range.
    """

    model_size: int = 1024
    key_value_size: int = 64
    num_heads: int = 16
    feedforward_size: int = 2816
    num_layers: int = 24
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-6
    input_length: int = 512
    patch_length: int = 8
    patch_dropout: float = 0.1
    head_dropout: float = 0.1
    normalization_epsilon: float = 1e-5
    is_positional_embedding_used: bool = True
    is_backbone_frozen: bool = False
    weights: PretrainedWeights | None = None

    def __post_init__(self) -> None:
        if self.input_length <= 0:
            raise ValueError(f"input_length must be positive, got {self.input_length}")
        if self.patch_length <= 0 or self.input_length % self.patch_length != 0:
            raise ValueError(f"patch_length must divide input_length {self.input_length}, got {self.patch_length}")
        for name, value in (("patch_dropout", self.patch_dropout), ("head_dropout", self.head_dropout)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.normalization_epsilon <= 0.0:
            raise ValueError(f"normalization_epsilon must be positive, got {self.normalization_epsilon}")
        # The encoder configuration validates the remaining sizes and the dropout.
        _ = self.encoder_configuration

    @classmethod
    def pretrained(cls, size: MOMENTSize) -> Self:
        """
        Get the architecture and the published weights of a MOMENT checkpoint.

        Parameters
        ----------
        size : MOMENTSize
            The checkpoint.

        Returns
        -------
        Self
            The parameters, whose networks load the checkpoint when built.
        """
        weights: PretrainedWeights = PretrainedWeights(repository=size.repository, revision=size.revision)
        match size:
            case MOMENTSize.SMALL:
                return cls(
                    model_size=512,
                    num_heads=6,
                    feedforward_size=1024,
                    num_layers=8,
                    is_backbone_frozen=True,
                    weights=weights,
                )
            case MOMENTSize.BASE:
                return cls(
                    model_size=768,
                    num_heads=12,
                    feedforward_size=2048,
                    num_layers=12,
                    is_backbone_frozen=True,
                    weights=weights,
                )
            case MOMENTSize.LARGE:
                return cls(is_backbone_frozen=True, weights=weights)

    @property
    def num_patches(self) -> int:
        """
        Get the number of patches per series.

        Returns
        -------
        int
            ``input_length // patch_length``.
        """
        return self.input_length // self.patch_length

    @property
    def encoder_configuration(self) -> T5Configuration:
        """
        Get the sizes of the T5 encoder.

        Returns
        -------
        T5Configuration
            The configuration.
        """
        return T5Configuration(
            model_size=self.model_size,
            key_value_size=self.key_value_size,
            num_heads=self.num_heads,
            feedforward_size=self.feedforward_size,
            num_layers=self.num_layers,
            feed_forward_kind=T5FeedForwardKind.GATED_GELU,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
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
        Build a forecast network with a new head, loading the published encoder if ``weights`` is set.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network, which must be fine-tuned before it forecasts.

        Raises
        ------
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: MOMENTForecastNetwork = MOMENTForecastNetwork(self, shape)
        self._load(network.encoder, MOMENTOfficialNames.encoder_mapping())
        return network

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build a summary network with a new head, loading the published encoder if ``weights`` is set.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions; at most ``input_length`` steps.

        Returns
        -------
        SummaryNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences are longer than ``input_length``.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: MOMENTSummaryNetwork = MOMENTSummaryNetwork(self, shape)
        self._load(network.encoder, MOMENTOfficialNames.encoder_mapping())
        return network

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build a reconstruction network, loading the published encoder and reconstruction head if ``weights`` is set.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences; at most ``input_length`` steps.

        Returns
        -------
        ReconstructionNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences are longer than ``input_length``.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: MOMENTReconstructionNetwork = MOMENTReconstructionNetwork(self, shape)
        self._load(network, MOMENTOfficialNames.reconstruction_mapping())
        return network

    def build_masked_reconstruction_network(self, shape: SequenceShape) -> MaskedReconstructionNetwork:
        """
        Build a mask-aware reconstruction network for imputation, loading the published encoder and reconstruction
        head if ``weights`` is set.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences; at most ``input_length`` steps.

        Returns
        -------
        MaskedReconstructionNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences are longer than ``input_length``.
        MissingDependencyError
            If weights are set but the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the published weights do not fit this architecture.
        """
        network: MOMENTMaskedReconstructionNetwork = MOMENTMaskedReconstructionNetwork(self, shape)
        self._load(network, MOMENTOfficialNames.reconstruction_mapping())
        return network

    def _load(self, module: nn.Module, mapping: ParameterNameMapping) -> None:
        if self.weights is not None:
            PretrainedWeightsLoader.load(module, self.weights, mapping)
