from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers import TemporalPooling
from ....interfaces.pretraining_network import PretrainingNetwork
from .augmentation import TSTCCAugmentation
from .contextual_contrastive_loss import ContextualContrastiveLoss
from .sequence_encoder import TSTCCSequenceEncoder
from .temporal_contrasting import TSTCCTemporalContrasting


class TSTCCPretrainingNetwork(PretrainingNetwork):
    """
    TS-TCC pretraining: the weak and strong augmentations of each sequence are encoded and normalized, the context
    of each view predicts the future features of the other (temporal contrasting), and the projected contexts of the
    two views are contrasted across the batch (contextual contrasting).

    Parameters
    ----------
    encoder : TSTCCSequenceEncoder
        The encoder to pretrain.
    temporal_contrasting : TSTCCTemporalContrasting
        The context Transformer and the predictors of future features.
    augmentation : TSTCCAugmentation
        The weak and strong augmentations.
    contextual_loss : ContextualContrastiveLoss
        The contrastive loss of the projected contexts.
    context_size : int
        Size of the context vectors.
    temporal_weight : float
        Weight of the two temporal contrasting losses.
    contextual_weight : float
        Weight of the contextual contrasting loss.

    Raises
    ------
    ValueError
        If the encoder's output is not longer than the number of predicted steps, or its feature size is below 4.
    """

    def __init__(
        self,
        encoder: TSTCCSequenceEncoder,
        temporal_contrasting: TSTCCTemporalContrasting,
        augmentation: TSTCCAugmentation,
        contextual_loss: ContextualContrastiveLoss,
        context_size: int,
        temporal_weight: float,
        contextual_weight: float,
    ) -> None:
        if encoder.output_length <= temporal_contrasting.num_steps:
            raise ValueError(
                f"the encoder's output of {encoder.output_length} steps must be longer than the "
                + f"{temporal_contrasting.num_steps} predicted steps; use longer sequences or fewer prediction_steps"
            )
        if encoder.output_size < 4:
            raise ValueError(f"the feature size must be at least 4 for the projection, got {encoder.output_size}")
        super().__init__()
        self.sequence_encoder: TSTCCSequenceEncoder = encoder
        self.temporal_contrasting: TSTCCTemporalContrasting = temporal_contrasting
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(context_size, encoder.output_size // 2),
            nn.BatchNorm1d(encoder.output_size // 2),
            nn.ReLU(),
            nn.Linear(encoder.output_size // 2, encoder.output_size // 4),
        )
        self.augmentation: TSTCCAugmentation = augmentation
        self.contextual_loss: ContextualContrastiveLoss = contextual_loss
        self.temporal_weight: float = temporal_weight
        self.contextual_weight: float = contextual_weight

    @property
    @override
    def encoder(self) -> TSTCCSequenceEncoder:
        return self.sequence_encoder

    @property
    @override
    def pooling(self) -> TemporalPooling:
        return TemporalPooling.FLATTEN

    @override
    def forward(self, values: Tensor) -> Tensor:
        weak: Tensor = functional.normalize(self.sequence_encoder(self.augmentation.weak(values)), dim=-1)
        strong: Tensor = functional.normalize(self.sequence_encoder(self.augmentation.strong(values)), dim=-1)
        weak_loss: Tensor
        weak_context: Tensor
        weak_loss, weak_context = self.temporal_contrasting(weak, strong)
        strong_loss: Tensor
        strong_context: Tensor
        strong_loss, strong_context = self.temporal_contrasting(strong, weak)
        # Both views share one batch normalization pass, so a batch of one sequence still has two samples.
        projected: Tensor = self.projection(torch.cat([weak_context, strong_context], dim=0))
        weak_projected: Tensor = projected[: values.shape[0]]
        strong_projected: Tensor = projected[values.shape[0] :]
        return self.temporal_weight * (weak_loss + strong_loss) + self.contextual_weight * self.contextual_loss.compute(
            weak_projected, strong_projected
        )
