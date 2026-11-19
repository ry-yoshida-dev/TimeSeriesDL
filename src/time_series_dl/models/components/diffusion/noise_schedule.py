from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .diffusion_process import DiffusionProcess
from .noise_schedule_kind import NoiseScheduleKind


@dataclass(frozen=True, kw_only=True)
class NoiseSchedule:
    """
    Noise variances of the forward process of a denoising diffusion model.

    Attributes
    ----------
    kind : NoiseScheduleKind
        How the variances grow from ``first_variance`` to ``last_variance``.
    num_steps : int
        Number of diffusion steps, also the number of denoising steps when sampling.
    first_variance : float
        Noise variance (beta) of the first step.
    last_variance : float
        Noise variance (beta) of the last step.

    Raises
    ------
    ValueError
        If num_steps is not positive, or the variances are not in (0, 1) with the first at most the last.
    """

    kind: NoiseScheduleKind = NoiseScheduleKind.LINEAR
    num_steps: int = 100
    first_variance: float = 1e-4
    last_variance: float = 0.1

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if not 0.0 < self.first_variance < 1.0 or not 0.0 < self.last_variance < 1.0:
            raise ValueError(f"variances must be in (0, 1), got {self.first_variance} and {self.last_variance}")
        if self.first_variance > self.last_variance:
            raise ValueError(
                f"first_variance must not exceed last_variance, got {self.first_variance} > {self.last_variance}"
            )

    def compute_variances(self) -> Tensor:
        """
        Compute the noise variance of every step.

        Returns
        -------
        Tensor
            The variances in double precision with shape (num_steps,).
        """
        return self.kind.compute_variances(self.num_steps, self.first_variance, self.last_variance)

    def build_process(self) -> DiffusionProcess:
        """
        Build the diffusion process of this schedule.

        Returns
        -------
        DiffusionProcess
            The process.
        """
        return DiffusionProcess(self.compute_variances())
