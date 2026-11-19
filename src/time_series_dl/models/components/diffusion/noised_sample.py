from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class NoisedSample:
    """
    Clean values diffused to random steps, with the noise a denoiser learns to predict.

    Attributes
    ----------
    noisy : Tensor
        The diffused values, with the shape of the clean values.
    noise : Tensor
        The standard normal noise that was added, with the same shape.
    steps : Tensor
        The zero-based diffusion step of each sample with shape (batch,).
    """

    noisy: Tensor
    noise: Tensor
    steps: Tensor
