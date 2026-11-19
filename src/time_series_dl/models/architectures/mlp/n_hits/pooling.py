from enum import Enum

from torch import nn


class NHITSPooling(Enum):
    """
    Downsampling of the input window of an N-HiTS block.

    Attributes
    ----------
    MAX
        Maximum of each non-overlapping window, the paper's default.
    AVERAGE
        Mean of each non-overlapping window.
    """

    MAX = "max"
    AVERAGE = "average"

    def build(self, kernel_size: int) -> nn.Module:
        """
        Build the pooling layer over the last dimension; a partial last window is kept.

        Parameters
        ----------
        kernel_size : int
            Length of each pooled window, also its stride.

        Returns
        -------
        nn.Module
            The pooling layer, mapping a length L to ceil(L / kernel_size).

        Raises
        ------
        ValueError
            If kernel_size is not positive.
        """
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        match self:
            case NHITSPooling.MAX:
                return nn.MaxPool1d(kernel_size, stride=kernel_size, ceil_mode=True)
            case NHITSPooling.AVERAGE:
                return nn.AvgPool1d(kernel_size, stride=kernel_size, ceil_mode=True, count_include_pad=False)
