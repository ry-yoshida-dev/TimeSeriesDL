from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ....components.diffusion import DiffusionProcess


@dataclass(frozen=True, kw_only=True)
class CosineNoiseSchedule:
    """
    Cosine noise schedule of Nichol and Dhariwal (2021), used by Diffusion-TS: the signal level ``alpha_bar`` follows
    ``cos^2((n / N + offset) / (1 + offset) * pi / 2)`` over the steps, and each variance is
    ``1 - alpha_bar_n / alpha_bar_{n-1}``, clipped at ``max_variance``.

    Attributes
    ----------
    num_steps : int
        Number of diffusion steps, also the number of denoising steps when sampling.
    offset : float
        Small offset that keeps the first variances away from zero.
    max_variance : float
        Upper bound of every variance, below 1 so the last steps keep a trace of the signal.

    Raises
    ------
    ValueError
        If num_steps is not positive, offset is not positive or max_variance is not in (0, 1).
    """

    num_steps: int = 500
    offset: float = 0.008
    max_variance: float = 0.999

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.offset <= 0.0:
            raise ValueError(f"offset must be positive, got {self.offset}")
        if not 0.0 < self.max_variance < 1.0:
            raise ValueError(f"max_variance must be in (0, 1), got {self.max_variance}")

    def compute_variances(self) -> Tensor:
        """
        Compute the noise variance of every step.

        Returns
        -------
        Tensor
            The variances in double precision with shape (num_steps,).
        """
        positions: Tensor = torch.arange(self.num_steps + 1, dtype=torch.float64) / self.num_steps
        signal_levels: Tensor = torch.cos((positions + self.offset) / (1.0 + self.offset) * math.pi / 2.0).square()
        variances: Tensor = 1.0 - signal_levels[1:] / signal_levels[:-1]
        return variances.clamp(min=torch.finfo(torch.float64).tiny, max=self.max_variance)

    def build_process(self) -> DiffusionProcess:
        """
        Build the diffusion process of this schedule.

        Returns
        -------
        DiffusionProcess
            The process.
        """
        return DiffusionProcess(self.compute_variances())
