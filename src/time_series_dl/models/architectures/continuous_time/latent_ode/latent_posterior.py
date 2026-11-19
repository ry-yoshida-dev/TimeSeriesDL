from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class LatentPosterior:
    """
    Diagonal Gaussian posterior of the initial latent state of each sample.

    Attributes
    ----------
    mean : Tensor
        Means with shape (batch, latent_size).
    std : Tensor
        Positive standard deviations with the shape of ``mean``.
    """

    mean: Tensor
    std: Tensor

    def sample(self, is_random: bool) -> Tensor:
        """
        Draw latent states by reparametrization, or take the means.

        Parameters
        ----------
        is_random : bool
            Whether to draw random states instead of returning the means.

        Returns
        -------
        Tensor
            The latent states with the shape of ``mean``.
        """
        return self.mean + self.std * torch.randn_like(self.std) if is_random else self.mean

    def compute_kl_divergence(self) -> Tensor:
        """
        Compute the Kullback-Leibler divergence from the standard normal prior.

        Returns
        -------
        Tensor
            The divergence of each sample and latent dimension, with the shape of ``mean``.
        """
        variance: Tensor = self.std.square()
        return 0.5 * (self.mean.square() + variance - 1.0 - torch.log(variance))
