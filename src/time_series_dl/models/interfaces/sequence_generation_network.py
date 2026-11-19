from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class SequenceGenerationNetwork(nn.Module, ABC):
    """
    Generative model of fixed-length multivariate sequences, trained with its own objective, that draws new sequences
    from noise, such as the adversarially trained TimeGAN or the denoising diffusion model Diffusion-TS.

    Training minimizes ``compute_loss`` on scaled sequences with one optimizer over all parameters. Models trained in
    alternating or adversarial phases express them as one combined loss whose gradients equal those of the paper's
    phases, and may change the loss with the epoch. The value of the loss is what the trainer monitors for early
    stopping and for keeping the best epoch, so it may differ from the objective whose gradient it carries, as long
    as a lower value means a better generator.
    """

    @property
    def minimum_epochs(self) -> int:
        """
        Get the number of epochs training needs before the network generates sequences at all, such as the
        pretraining phases of a GAN followed by one adversarial epoch.

        Returns
        -------
        int
            The number of epochs; 1 unless a network says otherwise.
        """
        return 1

    @abstractmethod
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        """
        Compute the training loss of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Scaled sequences with shape (batch, length, channels).
        epoch : int
            Zero-based index of the current training epoch, for losses whose phases change during training.

        Returns
        -------
        Tensor
            The scalar loss.
        """

    @abstractmethod
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """
        Draw new scaled sequences.

        Parameters
        ----------
        num_samples : int
            Number of sequences.
        generator : torch.Generator | None
            Source of randomness on the device of the network; PyTorch's global generator when None, so a seeded
            generator makes sampling reproducible.

        Returns
        -------
        Tensor
            The sequences with shape (num_samples, length, channels).
        """
