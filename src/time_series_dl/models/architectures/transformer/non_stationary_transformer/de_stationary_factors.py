from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class DeStationaryFactors:
    """
    De-stationary factors that rescale and shift the attention scores computed from stationarized series, so that
    they approximate the scores of the raw series.

    Attributes
    ----------
    scale : Tensor
        The positive scaling factor tau with shape (batch, 1).
    shift : Tensor
        The shifting factor delta of every key position of the encoder with shape (batch, input_length).
    """

    scale: Tensor
    shift: Tensor
