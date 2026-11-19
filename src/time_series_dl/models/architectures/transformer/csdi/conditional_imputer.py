from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ....components.diffusion import DiffusionProcess, NoisedSample
from .denoiser import CSDIDenoiser

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDIConditionalImputer(nn.Module):
    """
    Mask-aware core of CSDI: the denoising loss of the target entries of (features, time) arrays and the sampling of
    every unconditioned entry, given which entries are conditioned on.

    Forecasting conditions on the past window and targets the horizon; imputation conditions on a random subset of
    the observed entries while training and on all of them when imputing. Masks hold ones and zeros in the dtype of
    the values.

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration.
    num_features : int
        Number of features of the arrays.
    """

    def __init__(self, parameters: CSDIParameters, num_features: int) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.denoiser: CSDIDenoiser = CSDIDenoiser(parameters, num_features)
        self.process: DiffusionProcess = parameters.noise_schedule.build_process()

    def compute_loss(self, values: Tensor, observed_mask: Tensor, conditional_mask: Tensor) -> Tensor:
        """
        Compute the denoising loss of the observed entries that are not conditioned on.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, features, length); unobserved entries may hold anything finite.
        observed_mask : Tensor
            One for observed entries and zero otherwise, with the same shape.
        conditional_mask : Tensor
            One for entries the model conditions on, a subset of the observed ones, with the same shape.

        Returns
        -------
        Tensor
            The mean squared error of the predicted noise over the target entries; zero without target entries.

        Raises
        ------
        ValueError
            If the shapes differ or the number of features is not ``num_features``.
        """
        self._check_shapes(values, observed_mask, conditional_mask)
        noised: NoisedSample = self.process.corrupt(values)
        predicted: Tensor = self.denoiser(
            conditional_mask * values, (1.0 - conditional_mask) * noised.noisy, conditional_mask, noised.steps
        )
        target_mask: Tensor = (observed_mask - conditional_mask).clamp_min(0.0)
        squared: Tensor = ((noised.noise - predicted) * target_mask).square()
        return squared.sum() / target_mask.sum().clamp_min(1.0)

    def impute(
        self, values: Tensor, conditional_mask: Tensor, num_samples: int, generator: torch.Generator | None = None
    ) -> Tensor:
        """
        Draw samples of the unconditioned entries by the reverse diffusion.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, features, length); only the conditioned entries are read.
        conditional_mask : Tensor
            One for entries to condition on and zero for entries to sample, with the same shape.
        num_samples : int
            Number of samples per array.
        generator : torch.Generator | None
            Source of randomness on the device of the network; PyTorch's global generator when None.

        Returns
        -------
        Tensor
            Samples with shape (batch, num_samples, features, length) that keep the conditioned values.

        Raises
        ------
        ValueError
            If the shapes differ, the number of features is not ``num_features``, or num_samples is not positive.
        """
        self._check_shapes(values, conditional_mask, conditional_mask)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        batch: int = values.shape[0]
        repeated_mask: Tensor = conditional_mask.repeat_interleave(num_samples, dim=0)
        conditional_values: Tensor = repeated_mask * values.repeat_interleave(num_samples, dim=0)
        unconditioned: Tensor = 1.0 - repeated_mask

        def predict_noise(noisy: Tensor, steps: Tensor) -> Tensor:
            return self.denoiser(conditional_values, unconditioned * noisy, repeated_mask, steps)

        drawn: Tensor = self.process.sample(predict_noise, tuple(conditional_values.shape), generator)
        imputed: Tensor = conditional_values + unconditioned * drawn.to(values.dtype)
        return imputed.reshape(batch, num_samples, *values.shape[1:])

    def _check_shapes(self, values: Tensor, first_mask: Tensor, second_mask: Tensor) -> None:
        if values.dim() != 3 or values.shape[1] != self.num_features:
            raise ValueError(f"values must have shape (batch, {self.num_features}, length), got {tuple(values.shape)}")
        if first_mask.shape != values.shape or second_mask.shape != values.shape:
            raise ValueError(f"masks must have the shape of the values {tuple(values.shape)}")
