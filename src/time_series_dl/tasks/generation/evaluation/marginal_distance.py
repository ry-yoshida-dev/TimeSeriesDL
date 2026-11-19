from dataclasses import dataclass

import numpy as np

from ....array_types import FloatArray


@dataclass(frozen=True)
class MarginalDistance:
    """
    Distance between the marginal distributions of the values of each channel, pooled over sequences and steps: the
    mean absolute difference of evenly spaced quantiles, which approximates the 1-Wasserstein distance, averaged over
    the channels.

    Attributes
    ----------
    num_quantiles : int
        Number of quantiles compared per channel, at the levels ``(i + 0.5) / num_quantiles``.
    """

    num_quantiles: int

    def compute(self, real: FloatArray, synthetic: FloatArray) -> float:
        """
        Compute the distance between real and synthetic sequences.

        Parameters
        ----------
        real : FloatArray
            Real sequences with shape (num_real, length, channels).
        synthetic : FloatArray
            Synthetic sequences with the same number of channels.

        Returns
        -------
        float
            The distance, in the units of the values; 0 for identical marginals.

        Raises
        ------
        ValueError
            If the numbers of channels differ.
        """
        if real.shape[-1] != synthetic.shape[-1]:
            raise ValueError(f"real and synthetic sequences differ in channels: {real.shape} and {synthetic.shape}")
        levels: FloatArray = (np.arange(self.num_quantiles) + 0.5) / self.num_quantiles
        real_quantiles: FloatArray = np.quantile(real.reshape(-1, real.shape[-1]), levels, axis=0)
        synthetic_quantiles: FloatArray = np.quantile(synthetic.reshape(-1, synthetic.shape[-1]), levels, axis=0)
        return float(np.mean(np.abs(real_quantiles - synthetic_quantiles)))
