from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....array_types import BoolArray, FloatArray


@dataclass(frozen=True)
class AutocorrelationDistance:
    """
    Distance between the temporal dependence of real and synthetic sequences: the mean absolute difference of their
    mean autocorrelation functions over lags ``1..max_lag`` and channels.

    The autocorrelation of a sequence's channel at lag ``k`` is ``sum_t (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2``
    with its own mean ``m``; constant channels count as uncorrelated.

    Attributes
    ----------
    max_lag : int | None
        Largest lag; ``length - 1`` when None or larger.
    """

    max_lag: int | None

    def compute(self, real: FloatArray, synthetic: FloatArray) -> float:
        """
        Compute the distance between real and synthetic sequences.

        Parameters
        ----------
        real : FloatArray
            Real sequences with shape (num_real, length, channels), at least two steps long.
        synthetic : FloatArray
            Synthetic sequences with the same length and channels.

        Returns
        -------
        float
            The distance in [0, 2]; 0 for identical mean autocorrelations.

        Raises
        ------
        ValueError
            If the shapes differ or the sequences are shorter than two steps.
        """
        if real.shape[1:] != synthetic.shape[1:]:
            raise ValueError(f"real and synthetic sequences differ in shape: {real.shape} and {synthetic.shape}")
        length: int = int(real.shape[1])
        if length < 2:
            raise ValueError(f"the autocorrelation distance needs sequences of at least 2 steps, got {length}")
        num_lags: int = length - 1 if self.max_lag is None else min(self.max_lag, length - 1)
        gap: FloatArray = self._compute_mean_autocorrelation(real, num_lags) - self._compute_mean_autocorrelation(
            synthetic, num_lags
        )
        return float(np.mean(np.abs(gap)))

    @staticmethod
    def _compute_mean_autocorrelation(values: FloatArray, num_lags: int) -> FloatArray:
        # Autocovariances of every lag at once from the power spectrum, zero-padded against circular wrap-around.
        length: int = int(values.shape[1])
        centered: FloatArray = values.astype(np.float64) - values.mean(axis=1, keepdims=True, dtype=np.float64)
        spectrum: NDArray[np.complexfloating] = np.fft.rfft(centered, n=2 * length, axis=1)
        covariances: FloatArray = np.fft.irfft(np.abs(spectrum) ** 2, n=2 * length, axis=1)[:, : num_lags + 1]
        variances: FloatArray = covariances[:, :1]
        is_constant: BoolArray = variances <= np.finfo(np.float64).eps * length
        correlations: FloatArray = np.where(
            is_constant, 0.0, covariances[:, 1:] / np.where(is_constant, 1.0, variances)
        )
        return correlations.mean(axis=0)
