from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor

from ....components.diffusion import DiffusionProcess, NoisedSample
from ....interfaces import SequenceShape
from ....interfaces.sequence_generation_network import SequenceGenerationNetwork
from .denoiser import DiffusionTSDenoiser
from .loss_kind import DiffusionTSLossKind

if TYPE_CHECKING:
    from .parameters import DiffusionTSParameters


class DiffusionTSSequenceGenerationNetwork(SequenceGenerationNetwork):
    """
    Diffusion-TS: a denoising diffusion model whose interpretable Transformer denoiser estimates the clean sequences
    directly rather than the noise.

    Training diffuses each sequence to a uniformly drawn step and minimizes, per sequence, the mean error of the
    estimate in the time domain plus ``fourier_weight`` times the error of the real and imaginary parts of its
    Fourier transform (normalized by the length), weighted by the step's
    ``sqrt(1 - beta_n) sqrt(1 - alpha_bar_n) / beta_n / 100``. Sampling runs the ancestral sampler of the diffusion
    process from pure noise, turning each clean estimate into the noise it implies.

    Parameters
    ----------
    parameters : DiffusionTSParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: DiffusionTSParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.length: int = shape.length
        self.channels: int = shape.channels
        self.loss_kind: DiffusionTSLossKind = parameters.loss_kind
        self.fourier_weight: float = (
            shape.length**0.5 / 5.0 if parameters.fourier_weight is None else parameters.fourier_weight
        )
        self.denoiser: DiffusionTSDenoiser = DiffusionTSDenoiser(parameters, shape)
        self.process: DiffusionProcess = parameters.noise_schedule.build_process()
        betas: Tensor = self.process.betas
        self.loss_weights: Tensor
        self.register_buffer(
            "loss_weights", (1.0 - betas).sqrt() * self.process.noise_scales / betas / 100.0, persistent=False
        )

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        noised: NoisedSample = self.process.corrupt(values)
        return self.compute_denoising_loss(self.denoiser(noised.noisy, noised.steps), values, noised.steps)

    def compute_denoising_loss(self, estimate: Tensor, clean: Tensor, steps: Tensor) -> Tensor:
        """
        Compute the weighted time and frequency domain error of clean estimates.

        Parameters
        ----------
        estimate : Tensor
            Estimated clean sequences with shape (batch, length, channels).
        clean : Tensor
            True clean sequences with the same shape.
        steps : Tensor
            Zero-based diffusion step of each sequence with shape (batch,).

        Returns
        -------
        Tensor
            The scalar loss.
        """
        error: Tensor = self.loss_kind.compute(estimate - clean)
        spectrum_gap: Tensor = cast(Tensor, torch.fft.fft(estimate, dim=1, norm="forward")) - cast(
            Tensor, torch.fft.fft(clean, dim=1, norm="forward")
        )
        fourier_error: Tensor = self.loss_kind.compute(spectrum_gap.real) + self.loss_kind.compute(spectrum_gap.imag)
        per_sequence: Tensor = (error + self.fourier_weight * fourier_error).flatten(start_dim=1).mean(dim=1)
        return (per_sequence * self.loss_weights[steps]).mean()

    @override
    def sample(self, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        def predict_noise(noisy: Tensor, steps: Tensor) -> Tensor:
            estimate: Tensor = self.denoiser(noisy, steps)
            signal: Tensor = self.process.signal_scales[steps].view(-1, 1, 1)
            spread: Tensor = self.process.noise_scales[steps].view(-1, 1, 1)
            return (noisy - signal * estimate) / spread

        return self.process.sample(predict_noise, (num_samples, self.length, self.channels), generator)
