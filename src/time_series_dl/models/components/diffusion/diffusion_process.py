from collections.abc import Callable

import torch
from torch import Tensor, nn

from .noised_sample import NoisedSample

type NoisePredictor = Callable[[Tensor, Tensor], Tensor]
"""Maps noisy values and their zero-based diffusion steps with shape (batch,) to the predicted noise."""


class DiffusionProcess(nn.Module):
    """
    Forward noising and ancestral sampling of a denoising diffusion probabilistic model (Ho et al., 2020).

    The forward process diffuses clean values ``x_0`` to ``x_n = sqrt(alpha_bar_n) x_0 + sqrt(1 - alpha_bar_n) noise``
    at step ``n``. A denoiser learns to predict the noise, and sampling runs the reverse process from pure noise with
    the posterior variance ``(1 - alpha_bar_{n-1}) / (1 - alpha_bar_n) beta_n`` at every step but the last.

    The schedule is held in non-persistent buffers, so it moves with the network and is not saved with its weights.

    Parameters
    ----------
    variances : Tensor
        The noise variance (beta) of every step with shape (num_steps,), each in (0, 1).

    Raises
    ------
    ValueError
        If the variances are empty, not one-dimensional or out of (0, 1).
    """

    def __init__(self, variances: Tensor) -> None:
        if variances.dim() != 1 or variances.numel() == 0:
            raise ValueError(f"variances must be a non-empty vector, got shape {tuple(variances.shape)}")
        if bool((variances <= 0.0).any()) or bool((variances >= 1.0).any()):
            raise ValueError("variances must be in (0, 1).")
        super().__init__()
        betas: Tensor = variances.double()
        alphas: Tensor = 1.0 - betas
        alpha_bars: Tensor = torch.cumprod(alphas, dim=0)
        previous_alpha_bars: Tensor = torch.cat([alpha_bars.new_ones(1), alpha_bars[:-1]])
        posterior_variances: Tensor = (1.0 - previous_alpha_bars) / (1.0 - alpha_bars) * betas
        dtype: torch.dtype = torch.get_default_dtype()
        self.betas: Tensor
        self.alpha_bars: Tensor
        self.signal_scales: Tensor
        self.noise_scales: Tensor
        self.reciprocal_alpha_roots: Tensor
        self.noise_coefficients: Tensor
        self.posterior_deviations: Tensor
        for name, value in (
            ("betas", betas),
            ("alpha_bars", alpha_bars),
            ("signal_scales", alpha_bars.sqrt()),
            ("noise_scales", (1.0 - alpha_bars).sqrt()),
            ("reciprocal_alpha_roots", alphas.rsqrt()),
            ("noise_coefficients", betas / (1.0 - alpha_bars).sqrt()),
            ("posterior_deviations", posterior_variances.sqrt()),
        ):
            self.register_buffer(name, value.to(dtype), persistent=False)

    @property
    def num_steps(self) -> int:
        """
        Get the number of diffusion steps.

        Returns
        -------
        int
            The number of steps.
        """
        return self.betas.shape[0]

    def diffuse(self, clean: Tensor, steps: Tensor, noise: Tensor) -> Tensor:
        """
        Diffuse clean values to the given steps of the forward process.

        Parameters
        ----------
        clean : Tensor
            Clean values with shape (batch, ...).
        steps : Tensor
            Zero-based diffusion step of each sample with shape (batch,).
        noise : Tensor
            Standard normal noise with the shape of the clean values.

        Returns
        -------
        Tensor
            The noisy values.

        Raises
        ------
        ValueError
            If the shapes do not match or a step is out of range.
        """
        if noise.shape != clean.shape:
            raise ValueError(f"noise must have shape {tuple(clean.shape)}, got {tuple(noise.shape)}")
        if steps.shape != (clean.shape[0],):
            raise ValueError(f"steps must have shape ({clean.shape[0]},), got {tuple(steps.shape)}")
        if bool((steps < 0).any()) or bool((steps >= self.num_steps).any()):
            raise ValueError(f"steps must be in [0, {self.num_steps}).")
        broadcast: tuple[int, ...] = (clean.shape[0],) + (1,) * (clean.dim() - 1)
        signal: Tensor = self.signal_scales[steps].reshape(broadcast).to(clean.dtype)
        spread: Tensor = self.noise_scales[steps].reshape(broadcast).to(clean.dtype)
        return signal * clean + spread * noise

    def corrupt(self, clean: Tensor, generator: torch.Generator | None = None) -> NoisedSample:
        """
        Diffuse each sample to a uniformly drawn step with fresh noise, for the denoising loss.

        Parameters
        ----------
        clean : Tensor
            Clean values with shape (batch, ...).
        generator : torch.Generator | None
            Source of randomness on the device of the values; PyTorch's global generator when None.

        Returns
        -------
        NoisedSample
            The noisy values, the noise and the steps.
        """
        steps: Tensor = torch.randint(0, self.num_steps, (clean.shape[0],), generator=generator, device=clean.device)
        noise: Tensor = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
        return NoisedSample(noisy=self.diffuse(clean, steps, noise), noise=noise, steps=steps)

    def sample(
        self, predict_noise: NoisePredictor, shape: tuple[int, ...], generator: torch.Generator | None = None
    ) -> Tensor:
        """
        Draw samples by running the reverse process from standard normal noise.

        Parameters
        ----------
        predict_noise : NoisePredictor
            The denoiser, conditioned on whatever the samples depend on.
        shape : tuple[int, ...]
            Shape of the samples, starting with the batch.
        generator : torch.Generator | None
            Source of randomness on the device of the process; PyTorch's global generator when None, so a seeded
            generator makes sampling reproducible.

        Returns
        -------
        Tensor
            The samples with the given shape.

        Raises
        ------
        ValueError
            If the shape is empty or has a non-positive dimension.
        """
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"shape must have positive dimensions, got {shape}")
        device: torch.device = self.betas.device
        dtype: torch.dtype = self.betas.dtype
        values: Tensor = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        for step in reversed(range(self.num_steps)):
            steps: Tensor = torch.full((shape[0],), step, device=device, dtype=torch.long)
            noise: Tensor = predict_noise(values, steps)
            values = self.reciprocal_alpha_roots[step] * (values - self.noise_coefficients[step] * noise)
            if step > 0:
                fresh: Tensor = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                values = values + self.posterior_deviations[step] * fresh
        return values
