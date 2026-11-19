import math
from typing import cast

from torch import Tensor, nn
from torch.nn import functional


class GatedFeedForward(nn.Module):
    """
    Gated feedforward network of the sLSTM block (post up-projection): a GELU-gated up-projection followed by a
    down-projection.

    The hidden size is ``factor * size`` rounded up to a multiple of 64, as in the official code.

    Parameters
    ----------
    size : int
        Size of the input and output vectors.
    factor : float
        Ratio of the hidden size to the size before rounding.
    dropout : float
        Dropout probability of the output.

    Raises
    ------
    ValueError
        If size or factor is not positive, or dropout is not in [0, 1).
    """

    ROUNDING: int = 64

    def __init__(self, size: int, factor: float, dropout: float) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if factor <= 0.0:
            raise ValueError(f"factor must be positive, got {factor}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        hidden_size: int = math.ceil(factor * size / self.ROUNDING) * self.ROUNDING
        self.up_projection: nn.Linear = nn.Linear(size, 2 * hidden_size, bias=False)
        self.down_projection: nn.Linear = nn.Linear(hidden_size, size, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the network.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``size``.

        Returns
        -------
        Tensor
            Outputs with the shape of ``values``.
        """
        gate, projected = self.up_projection(values).chunk(2, dim=-1)
        outputs: Tensor = self.dropout(self.down_projection(functional.gelu(gate) * projected))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
