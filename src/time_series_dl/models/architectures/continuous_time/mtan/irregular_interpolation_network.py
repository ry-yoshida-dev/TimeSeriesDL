import math
from typing import override

import torch
from torch import Tensor, nn

from ....components.layers import MultilayerPerceptron
from ....interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....interfaces.irregular_tensors import IrregularTensors
from .multi_time_attention import MultiTimeAttention
from .time_embedding import LearnedTimeEmbedding


class MTANInterpolationNetwork(IrregularInterpolationNetwork):
    """
    mTAND-Full (Shukla and Marlin, 2021): a variational encoder-decoder whose latent states sit at fixed reference
    times in [0, 1].

    The encoder attends from the reference times to the observed values and masks, reads the result with a
    bidirectional GRU and outputs a Gaussian posterior of each reference latent state. The decoder reads the latent
    states with a bidirectional GRU and attends from the query times to the reference times. Training minimizes the
    negative evidence lower bound with a Gaussian likelihood of fixed standard deviation, normalized by the number of
    observed targets of each sample; predictions decode the posterior means.

    Parameters
    ----------
    channels : int
        Number of channels of the sequences.
    hidden_size : int
        Size of the attention outputs and of the GRU states in each direction.
    latent_size : int
        Size of each reference latent state.
    projection_size : int
        Hidden size of the layers that output the posterior and the values.
    time_embedding_size : int
        Size of the learned time embeddings.
    num_heads : int
        Number of attention heads.
    num_reference_points : int
        Number of equally spaced reference times in [0, 1].
    observation_std : float
        Standard deviation of the Gaussian likelihood of the observed values.
    kl_weight : float
        Weight of the Kullback-Leibler divergence in the loss.

    Raises
    ------
    ValueError
        If a size is not positive, the embedding size is not divisible by the number of heads, observation_std is
        not positive or kl_weight is negative.
    """

    def __init__(
        self,
        channels: int,
        hidden_size: int,
        latent_size: int,
        projection_size: int,
        time_embedding_size: int,
        num_heads: int,
        num_reference_points: int,
        observation_std: float,
        kl_weight: float,
    ) -> None:
        if num_reference_points <= 0:
            raise ValueError(f"num_reference_points must be positive, got {num_reference_points}")
        if observation_std <= 0.0:
            raise ValueError(f"observation_std must be positive, got {observation_std}")
        if kl_weight < 0.0:
            raise ValueError(f"kl_weight must not be negative, got {kl_weight}")
        super().__init__()
        self.latent_size: int = latent_size
        self.observation_std: float = observation_std
        self.kl_weight: float = kl_weight
        self.reference_times: Tensor
        self.register_buffer("reference_times", torch.linspace(0.0, 1.0, num_reference_points))
        self.encoder_embedding: LearnedTimeEmbedding = LearnedTimeEmbedding(time_embedding_size)
        self.encoder_attention: MultiTimeAttention = MultiTimeAttention(
            2 * channels, hidden_size, time_embedding_size, num_heads
        )
        self.encoder_recurrence: nn.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.posterior: MultilayerPerceptron = MultilayerPerceptron(
            2 * hidden_size, (projection_size,), 2 * latent_size
        )
        self.decoder_embedding: LearnedTimeEmbedding = LearnedTimeEmbedding(time_embedding_size)
        self.decoder_recurrence: nn.GRU = nn.GRU(latent_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_attention: MultiTimeAttention = MultiTimeAttention(
            2 * hidden_size, 2 * hidden_size, time_embedding_size, num_heads
        )
        self.output_layers: MultilayerPerceptron = MultilayerPerceptron(2 * hidden_size, (projection_size,), channels)

    @override
    def forward(self, context: IrregularTensors, query_times: Tensor) -> Tensor:
        mean: Tensor = self._encode(context)[0]
        return self._decode(mean, query_times)

    @override
    def compute_training_loss(self, context: IrregularTensors, targets: IrregularTensors) -> Tensor:
        mean, log_variance = self._encode(context)
        std: Tensor = torch.exp(0.5 * log_variance)
        latent: Tensor = mean + std * torch.randn_like(std) if self.training else mean
        predictions: Tensor = self._decode(latent, targets.times)
        mask: Tensor = targets.observed_mask
        negative_log_likelihood: Tensor = (
            0.5 * ((predictions - targets.values) / self.observation_std).square()
            + math.log(self.observation_std)
            + 0.5 * math.log(2.0 * math.pi)
        ) * mask
        divergence: Tensor = 0.5 * (mean.square() + std.square() - 1.0 - log_variance)
        counts: Tensor = mask.sum(dim=(1, 2))
        per_sample: Tensor = (
            negative_log_likelihood.sum(dim=(1, 2)) + self.kl_weight * divergence.sum(dim=(1, 2))
        ) / counts.clamp(min=1.0)
        is_counted: Tensor = (counts > 0).to(per_sample.dtype)
        return (per_sample * is_counted).sum() / is_counted.sum().clamp(min=1.0)

    def _encode(self, context: IrregularTensors) -> tuple[Tensor, Tensor]:
        mask: Tensor = context.observed_mask
        attended: Tensor = self.encoder_attention(
            self.encoder_embedding(self.reference_times).unsqueeze(0),
            self.encoder_embedding(context.times),
            torch.cat([context.values, mask], dim=-1),
            torch.cat([mask, mask], dim=-1),
        )
        posterior: Tensor = self.posterior(self.encoder_recurrence(attended)[0])
        return posterior[..., : self.latent_size], posterior[..., self.latent_size :]

    def _decode(self, latent: Tensor, query_times: Tensor) -> Tensor:
        states: Tensor = self.decoder_recurrence(latent)[0]
        attended: Tensor = self.decoder_attention(
            self.decoder_embedding(query_times),
            self.decoder_embedding(self.reference_times).unsqueeze(0),
            states,
        )
        outputs: Tensor = self.output_layers(attended)
        return outputs
