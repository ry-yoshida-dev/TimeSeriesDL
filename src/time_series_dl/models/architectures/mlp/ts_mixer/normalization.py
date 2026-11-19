from enum import Enum

from torch import nn

from .batch_normalization import TSMixerBatchNormalization


class TSMixerNormalization(Enum):
    """
    Normalization in front of each mixing MLP of TSMixer.

    Attributes
    ----------
    BATCH
        Batch normalization of each channel over the batch and time, the default of the paper.
    LAYER
        Layer normalization of each sequence over time and channels.
    """

    BATCH = "batch"
    LAYER = "layer"

    def build(self, length: int, channels: int) -> nn.Module:
        """
        Build the normalization of sequences with shape (batch, length, channels).

        Parameters
        ----------
        length : int
            Number of time steps.
        channels : int
            Number of channels.

        Returns
        -------
        nn.Module
            The normalization, which keeps the shape of its input.
        """
        match self:
            case TSMixerNormalization.BATCH:
                return TSMixerBatchNormalization(channels)
            case TSMixerNormalization.LAYER:
                return nn.LayerNorm([length, channels])
