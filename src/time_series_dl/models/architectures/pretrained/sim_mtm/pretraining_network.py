from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers import TemporalPooling
from ....components.masked_modeling import Masking, SequenceDecoder
from ....interfaces import SequenceEncoder, SequenceShape
from ....interfaces.pretraining_network import PretrainingNetwork
from .automatic_loss_weighting import SimMTMAutomaticLossWeighting


class SimMTMPretrainingNetwork(PretrainingNetwork):
    """
    SimMTM pretraining around any sequence encoder: each sequence is masked several times, the series-wise
    similarities of all original and masked sequences weight the aggregation of the point-wise features of the
    other sequences, and the aggregate of each original sequence is decoded back to its values. A constraint loss
    makes the masked versions of a sequence its most similar neighbors.

    Parameters
    ----------
    encoder : SequenceEncoder
        The encoder to pretrain.
    shape : SequenceShape
        The dimensions of the unlabeled sequences.
    masking : Masking
        The random masks of the masked versions.
    num_masked : int
        Number of masked versions of each sequence.
    temperature : float
        Temperature of the similarities, in the constraint loss and the aggregation weights.
    projection_size : int
        Size of the series-wise representation used for the similarities.
    pooling : TemporalPooling
        Reduction of the point-wise features to the input of the series-wise projector, and to the instance-level
        representation.
    constraint_weight : float | None
        Fixed weight of the constraint loss added to the reconstruction loss; when None both losses are weighted by
        learned weights as in the official code.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        shape: SequenceShape,
        masking: Masking,
        num_masked: int,
        temperature: float,
        projection_size: int,
        pooling: TemporalPooling,
        constraint_weight: float | None,
    ) -> None:
        super().__init__()
        self.sequence_encoder: SequenceEncoder = encoder
        self.masking: Masking = masking
        self.num_masked: int = num_masked
        self.temperature: float = temperature
        self.feature_pooling: TemporalPooling = pooling
        self.constraint_weight: float = 0.0 if constraint_weight is None else constraint_weight
        pooled_size: int = pooling.output_size(encoder.output_length, encoder.output_size)
        hidden_size: int = max(pooled_size // 2, 1)
        self.projector: nn.Sequential = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size),
        )
        self.decoder: SequenceDecoder = SequenceDecoder(
            encoder.output_length, encoder.output_size, shape.length, shape.channels
        )
        self.loss_weighting: SimMTMAutomaticLossWeighting | None = (
            SimMTMAutomaticLossWeighting(2) if constraint_weight is None else None
        )

    @property
    @override
    def encoder(self) -> SequenceEncoder:
        return self.sequence_encoder

    @property
    @override
    def pooling(self) -> TemporalPooling:
        return self.feature_pooling

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch: int = values.shape[0]
        is_hidden: Tensor = self.masking.sample(
            batch * self.num_masked, values.shape[1], values.shape[2], values.device
        )
        masked: Tensor = values.repeat(self.num_masked, 1, 1).masked_fill(is_hidden, 0.0)
        features: Tensor = self.sequence_encoder(torch.cat([values, masked], dim=0))
        series: Tensor = functional.normalize(self.projector(self.feature_pooling.apply(features)), dim=1)
        similarity: Tensor = series @ series.T
        constraint_loss: Tensor = self.compute_constraint_loss(similarity, batch)
        reconstructed: Tensor = self.decoder(self.aggregate(similarity[:batch], features))
        reconstruction_loss: Tensor = functional.mse_loss(reconstructed, values)
        if self.loss_weighting is not None:
            return self.loss_weighting(torch.stack([constraint_loss, reconstruction_loss]))
        return reconstruction_loss + self.constraint_weight * constraint_loss

    def compute_constraint_loss(self, similarity: Tensor, batch: int) -> Tensor:
        """
        Compute the constraint loss: the masked versions of each sequence, and the sequence itself, are its
        positives among all other sequences.

        Parameters
        ----------
        similarity : Tensor
            Cosine similarities of the series-wise representations with shape (n, n), where n is a multiple of
            ``batch`` and item i is a version of sequence ``i % batch``.
        batch : int
            Number of original sequences.

        Returns
        -------
        Tensor
            The negative log-probabilities of the positives under the softmax of the temperature-scaled similarities
            over all other items, summed over the positives and averaged over the items.
        """
        count: int = similarity.shape[0]
        items: Tensor = torch.arange(count, device=similarity.device)
        is_self: Tensor = items.unsqueeze(1) == items.unsqueeze(0)
        sequences: Tensor = items % batch
        is_positive: Tensor = (sequences.unsqueeze(1) == sequences.unsqueeze(0)) & ~is_self
        log_probabilities: Tensor = functional.log_softmax(
            (similarity / self.temperature).masked_fill(is_self, -torch.inf), dim=1
        )
        return -log_probabilities.masked_fill(~is_positive, 0.0).sum(dim=1).mean()

    def aggregate(self, similarity: Tensor, features: Tensor) -> Tensor:
        """
        Aggregate the point-wise features of all other sequences, weighted by the softmax of their similarities.

        Parameters
        ----------
        similarity : Tensor
            Similarities of the first rows' items to all n items, with shape (rows, n); row i excludes item i.
        features : Tensor
            Point-wise features of all items with shape (n, feature_length, feature_size).

        Returns
        -------
        Tensor
            Aggregated features with shape (rows, feature_length, feature_size).
        """
        rows: int = similarity.shape[0]
        is_self: Tensor = torch.eye(rows, similarity.shape[1], dtype=torch.bool, device=similarity.device)
        weights: Tensor = torch.softmax((similarity / self.temperature).masked_fill(is_self, -torch.inf), dim=1)
        aggregated: Tensor = weights @ features.flatten(start_dim=1)
        return aggregated.reshape(rows, *features.shape[1:])
