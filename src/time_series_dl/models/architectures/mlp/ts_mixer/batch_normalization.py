from typing import cast

from torch import Tensor, nn


class TSMixerBatchNormalization(nn.Module):
    """
    Batch normalization of each channel of sequences laid out time first.

    Parameters
    ----------
    channels : int
        Number of channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(channels)

    def forward(self, values: Tensor) -> Tensor:
        """
        Normalize a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Normalized values with the same shape.
        """
        normalized: Tensor = self.normalization(values.transpose(1, 2))
        return normalized.transpose(1, 2)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
