from enum import Enum

from torch import Tensor


class SamplePointEstimate(Enum):
    """
    Statistic of sample paths that serves as the point forecast.

    Attributes
    ----------
    MEAN
        The sample mean, which minimizes the expected squared error.
    MEDIAN
        The sample median, which minimizes the expected absolute error (CSDI's choice).
    """

    MEAN = "mean"
    MEDIAN = "median"

    def compute(self, samples: Tensor) -> Tensor:
        """
        Compute the statistic over the last dimension.

        Parameters
        ----------
        samples : Tensor
            Samples along the last dimension.

        Returns
        -------
        Tensor
            The statistic, without the last dimension; the median of an even number of samples is the mean of the
            two middle ones.
        """
        match self:
            case SamplePointEstimate.MEAN:
                return samples.mean(dim=-1)
            case SamplePointEstimate.MEDIAN:
                ordered: Tensor = samples.sort(dim=-1).values
                middle: int = samples.shape[-1] // 2
                if samples.shape[-1] % 2 == 1:
                    return ordered[..., middle]
                return 0.5 * (ordered[..., middle - 1] + ordered[..., middle])
