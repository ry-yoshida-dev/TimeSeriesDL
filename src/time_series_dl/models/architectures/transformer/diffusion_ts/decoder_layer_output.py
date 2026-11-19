from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class DiffusionTSDecoderLayerOutput:
    """
    Outputs of one Diffusion-TS decoder layer.

    Attributes
    ----------
    tokens : Tensor
        The decoded tokens with their mean over time removed, with shape (batch, length, model_size).
    mean : Tensor
        The removed mean mapped to the output channels, with shape (batch, 1, output_channels).
    trend : Tensor
        The layer's polynomial trend with shape (batch, length, output_channels).
    seasonal : Tensor
        The layer's Fourier seasonality in token space with shape (batch, length, model_size).
    """

    tokens: Tensor
    mean: Tensor
    trend: Tensor
    seasonal: Tensor
