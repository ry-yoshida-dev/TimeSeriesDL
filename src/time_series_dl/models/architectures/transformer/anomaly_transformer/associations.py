from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class Associations:
    """
    Reconstruction and per-layer associations of a batch of windows computed by the Anomaly Transformer.

    Attributes
    ----------
    reconstruction : Tensor
        Reconstructed windows with shape (batch, length, channels).
    series : tuple[Tensor, ...]
        Series association of each layer with shape (batch, num_heads, length, length).
    prior : tuple[Tensor, ...]
        Row-normalized prior association of each layer with shape (batch, num_heads, length, length).
    """

    reconstruction: Tensor
    series: tuple[Tensor, ...]
    prior: tuple[Tensor, ...]
