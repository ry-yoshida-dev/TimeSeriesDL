from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class DiffusionTSDecomposition:
    """
    Interpretable parts of the clean sequences that the Diffusion-TS denoiser estimates; their sum is the estimate.

    Attributes
    ----------
    trend : Tensor
        Polynomial trend plus level with shape (batch, length, channels).
    seasonal : Tensor
        Sum of the Fourier components with shape (batch, length, channels).
    residual : Tensor
        Zero-mean remainder of the decoder tokens with shape (batch, length, channels).
    """

    trend: Tensor
    seasonal: Tensor
    residual: Tensor

    @property
    def estimate(self) -> Tensor:
        """
        Get the estimated clean sequences.

        Returns
        -------
        Tensor
            The sum of the parts with shape (batch, length, channels).
        """
        return self.trend + self.seasonal + self.residual
