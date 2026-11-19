from typing import cast

from torch import Tensor, nn


class GatedLinearOutput(nn.Module):
    """
    Output stage of a deep state space layer: GELU, dropout, and a linear map to twice the size halved by a gated
    linear unit.

    Parameters
    ----------
    size : int
        Size of the feature vectors.
    dropout : float
        Dropout probability after the activation.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(size, 2 * size), nn.GLU(dim=-1)
        )

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the output stage.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., size).

        Returns
        -------
        Tensor
            Outputs with the shape of ``values``.
        """
        outputs: Tensor = self.layers(values)
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
