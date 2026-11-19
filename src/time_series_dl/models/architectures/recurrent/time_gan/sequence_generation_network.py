from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ....components.adversarial import FrozenModule
from ....interfaces import SequenceShape
from ....interfaces.sequence_generation_network import SequenceGenerationNetwork
from .phase import TimeGANPhase
from .phase_schedule import TimeGANPhaseSchedule
from .recurrent_block import TimeGANRecurrentBlock

if TYPE_CHECKING:
    from .parameters import TimeGANParameters

# Loss weights and the variance offset of the moment loss, fixed in the authors' implementation.
RECONSTRUCTION_WEIGHT: Final[float] = 10.0
EMBEDDER_SUPERVISED_WEIGHT: Final[float] = 0.1
GENERATOR_SUPERVISED_WEIGHT: Final[float] = 100.0
MOMENT_WEIGHT: Final[float] = 100.0
VARIANCE_OFFSET: Final[float] = 1e-6


class TimeGANSequenceGenerationNetwork(SequenceGenerationNetwork):
    """
    TimeGAN: an embedder E and a recovery network R form an autoencoder into a latent space where a generator G maps
    noise Z to latent sequences, a supervisor S predicts the next latent step, and a discriminator D tells real latent
    sequences H = E(X) from generated ones. Samples are R(S(G(Z))).

    The authors train three phases with separate optimizers: (1) E and R minimize ``10 sqrt(MSE(X, R(H)))``; (2) S
    minimizes the supervised loss ``L_S = MSE(H[1:], S(H)[:-1])``; (3) jointly, G and S minimize
    ``BCE(D(S(G(Z))), 1) + gamma BCE(D(G(Z)), 1) + 100 sqrt(L_S) + 100 L_V`` with the moment loss ``L_V`` between the
    batch means and standard deviations of R(S(G(Z))) and X, E and R minimize ``10 sqrt(MSE(X, R(H))) + 0.1 L_S``,
    and D minimizes ``BCE(D(H), 1) + BCE(D(S(G(Z))), 0) + gamma BCE(D(G(Z)), 0)`` whenever that loss exceeds a
    threshold. The phase of an epoch comes from the ``TimeGANPhaseSchedule``.

    ``compute_loss`` is one loss per phase whose gradient equals the sum of the paper's phase gradients at the current
    weights: terms that must not train a sub-network see it frozen (``FrozenModule``) or its output detached. In the
    joint phase the paper takes two generator and embedder steps per discriminator step; here each batch takes one
    step of all of them.

    The loss's value is what the trainer monitors: before the joint phase, the number of epochs left until it plus a
    value in [0, 1) that grows with the phase's own loss, so every epoch before the joint phase counts as an
    improvement (they always run to the end, and are never kept as the best epoch); within the joint phase,
    ``L_V / (1 + L_V)``, so the epoch whose samples best match the batch moments is kept.

    The networks other than the discriminator end with a sigmoid, so the sequences should be scaled to [0, 1], as the
    generation task does by default.

    Parameters
    ----------
    parameters : TimeGANParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If the sequences are shorter than two steps, which the supervised loss needs.
    """

    def __init__(self, parameters: TimeGANParameters, shape: SequenceShape) -> None:
        if shape.length < 2:
            raise ValueError(f"TimeGAN needs sequences of at least 2 steps, got {shape.length}")
        super().__init__()
        self.length: int = shape.length
        self.noise_size: int = shape.channels if parameters.noise_size is None else parameters.noise_size
        self.schedule: TimeGANPhaseSchedule = parameters.schedule
        self.embedding_adversarial_weight: float = parameters.embedding_adversarial_weight
        self.discriminator_loss_threshold: float = parameters.discriminator_loss_threshold
        hidden_size: int = parameters.hidden_size
        num_layers: int = parameters.num_layers
        self.embedder: TimeGANRecurrentBlock = TimeGANRecurrentBlock(
            parameters.cell, shape.channels, hidden_size, num_layers, hidden_size, is_bounded=True
        )
        self.recovery: TimeGANRecurrentBlock = TimeGANRecurrentBlock(
            parameters.cell, hidden_size, hidden_size, num_layers, shape.channels, is_bounded=True
        )
        self.generator: TimeGANRecurrentBlock = TimeGANRecurrentBlock(
            parameters.cell, self.noise_size, hidden_size, num_layers, hidden_size, is_bounded=True
        )
        # The authors' supervisor has one layer less than the other networks.
        self.supervisor: TimeGANRecurrentBlock = TimeGANRecurrentBlock(
            parameters.cell, hidden_size, hidden_size, max(num_layers - 1, 1), hidden_size, is_bounded=True
        )
        self.discriminator: TimeGANRecurrentBlock = TimeGANRecurrentBlock(
            parameters.cell, hidden_size, hidden_size, num_layers, 1, is_bounded=False
        )

    @property
    @override
    def minimum_epochs(self) -> int:
        return self.schedule.first_joint_epoch + 1

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        noise: Tensor = torch.rand(
            values.shape[0], values.shape[1], self.noise_size, device=values.device, dtype=values.dtype
        )
        return self.compute_loss_with_noise(values, noise, epoch)

    def compute_loss_with_noise(self, values: Tensor, noise: Tensor, epoch: int) -> Tensor:
        """
        Compute the training loss of a batch of sequences from given generator inputs.

        Parameters
        ----------
        values : Tensor
            Scaled sequences with shape (batch, length, channels).
        noise : Tensor
            Generator inputs with shape (batch, length, noise_size); read only in the joint phase.
        epoch : int
            Zero-based index of the current training epoch, which selects the phase.

        Returns
        -------
        Tensor
            The scalar loss, with the gradient of the phase's objective and the monitored value.

        Raises
        ------
        ValueError
            If epoch is negative.
        """
        objective: Tensor
        monitored: Tensor
        match self.schedule.phase_at(epoch):
            case TimeGANPhase.AUTOENCODER:
                reconstruction: Tensor = functional.mse_loss(self.recovery(self.embedder(values)), values)
                objective = RECONSTRUCTION_WEIGHT * reconstruction.sqrt()
                monitored = reconstruction
            case TimeGANPhase.SUPERVISED:
                objective = self._compute_supervised_loss(self.embedder(values).detach(), self.supervisor)
                monitored = objective
            case TimeGANPhase.JOINT:
                objective, monitored = self._compute_joint_loss(values, noise)
        detached: Tensor = monitored.detach()
        value: Tensor = self.schedule.epochs_until_joint(epoch) + detached / (1.0 + detached)
        return objective - objective.detach() + value

    @override
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        reference: Tensor = self.recovery.output.weight
        noise: Tensor = torch.rand(
            num_samples,
            self.length,
            self.noise_size,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self.recovery(self.supervisor(self.generator(noise)))

    def _compute_joint_loss(self, values: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        latent: Tensor = self.embedder(values)
        real_latent: Tensor = latent.detach()
        generated: Tensor = self.generator(noise)
        supervised: Tensor = self.supervisor(generated)
        gamma: float = self.embedding_adversarial_weight
        # Generator and supervisor: fool the frozen discriminator, predict the next real latent step, match moments.
        frozen_discriminator: FrozenModule = FrozenModule(self.discriminator)
        generator_adversarial: Tensor = self._compute_cross_entropy(
            frozen_discriminator(supervised), True
        ) + gamma * self._compute_cross_entropy(frozen_discriminator(generated), True)
        synthetic: Tensor = FrozenModule(self.recovery)(supervised)
        moments: Tensor = self._compute_moment_loss(synthetic, values)
        generator_loss: Tensor = (
            generator_adversarial
            + GENERATOR_SUPERVISED_WEIGHT * self._compute_supervised_loss(real_latent, self.supervisor).sqrt()
            + MOMENT_WEIGHT * moments
        )
        # Embedder and recovery: reconstruct, and make the latent steps predictable by the frozen supervisor.
        reconstruction: Tensor = functional.mse_loss(self.recovery(latent), values)
        embedder_loss: Tensor = RECONSTRUCTION_WEIGHT * reconstruction.sqrt() + (
            EMBEDDER_SUPERVISED_WEIGHT * self._compute_supervised_loss(latent, FrozenModule(self.supervisor))
        )
        # Discriminator: tell real latent sequences from generated ones, trained only while it is not yet good.
        discriminator_loss: Tensor = (
            self._compute_cross_entropy(self.discriminator(real_latent), True)
            + self._compute_cross_entropy(self.discriminator(supervised.detach()), False)
            + gamma * self._compute_cross_entropy(self.discriminator(generated.detach()), False)
        )
        is_discriminator_trained: Tensor = (discriminator_loss.detach() > self.discriminator_loss_threshold).to(
            discriminator_loss.dtype
        )
        return generator_loss + embedder_loss + is_discriminator_trained * discriminator_loss, moments

    @staticmethod
    def _compute_supervised_loss(latent: Tensor, supervisor: TimeGANRecurrentBlock | FrozenModule) -> Tensor:
        return functional.mse_loss(supervisor(latent)[:, :-1, :], latent[:, 1:, :])

    @staticmethod
    def _compute_cross_entropy(logits: Tensor, is_real: bool) -> Tensor:
        targets: Tensor = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
        return functional.binary_cross_entropy_with_logits(logits, targets)

    @staticmethod
    def _compute_moment_loss(synthetic: Tensor, values: Tensor) -> Tensor:
        synthetic_variance: Tensor
        synthetic_mean: Tensor
        real_variance: Tensor
        real_mean: Tensor
        synthetic_variance, synthetic_mean = torch.var_mean(synthetic, dim=0, correction=0)
        real_variance, real_mean = torch.var_mean(values, dim=0, correction=0)
        deviation_gap: Tensor = (synthetic_variance + VARIANCE_OFFSET).sqrt() - (real_variance + VARIANCE_OFFSET).sqrt()
        return deviation_gap.abs().mean() + (synthetic_mean - real_mean).abs().mean()
