import math
from collections.abc import Sequence
from typing import Final

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.differential_equations import ODESolver
from ....components.layers import Activation, MultilayerPerceptron
from ....interfaces.irregular_tensors import IrregularTensors
from ..ode_rnn import ODERNNRecurrence
from .latent_posterior import LatentPosterior

_POSTERIOR_HIDDEN_SIZE: Final[int] = 100
_MINIMUM_STD: Final[float] = 1e-4


class LatentODEGenerativeModel(nn.Module):
    """
    Latent ODE (Rubanova et al., 2019): a latent state at time 0 follows a neural ordinary differential equation,
    and a linear map of the latent state gives the mean of the observations. An ODE-RNN read backward from the last
    observation to time 0 infers a Gaussian posterior of the initial latent state.

    Parameters
    ----------
    channels : int
        Number of channels of the sequences.
    latent_size : int
        Size of the latent state.
    recognition : ODERNNRecurrence
        The ODE-RNN of the posterior, read backward.
    dynamics_hidden_sizes : Sequence[int]
        Hidden layer sizes of the tanh network that defines the derivative of the latent state.
    solver : ODESolver
        Fixed-step solver of the latent dynamics.
    steps_per_interval : int
        Number of solver steps between consecutive output times.
    observation_std : float
        Standard deviation of the Gaussian likelihood of the observed values.
    kl_weight : float
        Weight of the Kullback-Leibler divergence in the loss.

    Raises
    ------
    ValueError
        If a size or steps_per_interval is not positive, observation_std is not positive or kl_weight is negative.
    """

    def __init__(
        self,
        channels: int,
        latent_size: int,
        recognition: ODERNNRecurrence,
        dynamics_hidden_sizes: Sequence[int],
        solver: ODESolver,
        steps_per_interval: int,
        observation_std: float,
        kl_weight: float,
    ) -> None:
        if steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {steps_per_interval}")
        if observation_std <= 0.0:
            raise ValueError(f"observation_std must be positive, got {observation_std}")
        if kl_weight < 0.0:
            raise ValueError(f"kl_weight must not be negative, got {kl_weight}")
        super().__init__()
        self.latent_size: int = latent_size
        self.solver: ODESolver = solver
        self.steps_per_interval: int = steps_per_interval
        self.observation_std: float = observation_std
        self.kl_weight: float = kl_weight
        self.recognition: ODERNNRecurrence = recognition
        self.posterior: MultilayerPerceptron = MultilayerPerceptron(
            recognition.hidden_size, (_POSTERIOR_HIDDEN_SIZE,), 2 * latent_size, Activation.TANH
        )
        self.dynamics: MultilayerPerceptron = MultilayerPerceptron(
            latent_size, dynamics_hidden_sizes, latent_size, Activation.TANH
        )
        self.output: nn.Linear = nn.Linear(latent_size, channels)

    def infer(self, sequences: IrregularTensors) -> LatentPosterior:
        """
        Infer the posterior of the initial latent state.

        Parameters
        ----------
        sequences : IrregularTensors
            The observed values.

        Returns
        -------
        LatentPosterior
            The posterior.
        """
        parameters: Tensor = self.posterior(self.recognition(sequences, is_backward=True))
        return LatentPosterior(
            mean=parameters[:, : self.latent_size],
            std=functional.softplus(parameters[:, self.latent_size :]) + _MINIMUM_STD,
        )

    def decode(self, latent: Tensor, times: Tensor) -> Tensor:
        """
        Follow the latent dynamics from time 0 and predict the mean of the observations.

        Parameters
        ----------
        latent : Tensor
            Initial latent states with shape (batch, latent_size).
        times : Tensor
            Non-decreasing output times of each sample with shape (batch, num_times).

        Returns
        -------
        Tensor
            The predicted means with shape (batch, num_times, channels).
        """
        states: Tensor = self.solver.integrate_through(
            lambda _, state: self.dynamics(state),
            latent,
            torch.zeros_like(times[:, :1]),
            times,
            self.steps_per_interval,
        )
        outputs: Tensor = self.output(states)
        return outputs

    def compute_negative_elbo(self, posterior: LatentPosterior, latent: Tensor, targets: IrregularTensors) -> Tensor:
        """
        Compute the negative evidence lower bound of observed targets: the Gaussian negative log-likelihood averaged
        over the observed values plus ``kl_weight`` times the Kullback-Leibler divergence averaged over the latent
        dimensions and the samples.

        Parameters
        ----------
        posterior : LatentPosterior
            The posterior of the initial latent state.
        latent : Tensor
            A latent state drawn from the posterior with shape (batch, latent_size).
        targets : IrregularTensors
            The values to reconstruct.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        predictions: Tensor = self.decode(latent, targets.times)
        mask: Tensor = targets.observed_mask
        negative_log_likelihood: Tensor = (
            0.5 * ((predictions - targets.values) / self.observation_std).square()
            + math.log(self.observation_std)
            + 0.5 * math.log(2.0 * math.pi)
        )
        reconstruction: Tensor = (negative_log_likelihood * mask).sum() / mask.sum().clamp(min=1.0)
        return reconstruction + self.kl_weight * posterior.compute_kl_divergence().mean()
