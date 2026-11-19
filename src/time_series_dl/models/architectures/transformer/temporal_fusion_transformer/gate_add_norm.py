from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class GateAddNorm(nn.Module):
    """
    Gated linear unit followed by a residual connection and layer normalization, the gated skip connection of TFT.

    Parameters
    ----------
    input_size : int
        Size of the gated inputs.
    output_size : int
        Size of the outputs and of the residual.
    dropout : float
        Dropout probability before the gate.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(self, input_size: int, output_size: int, dropout: float) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"input_size and output_size must be positive, got {input_size} and {output_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.gate: nn.Linear = nn.Linear(input_size, 2 * output_size)
        self.normalization: nn.LayerNorm = nn.LayerNorm(output_size)

    def forward(self, values: Tensor, residual: Tensor) -> Tensor:
        """
        Gate the values and add them to the residual.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``input_size``.
        residual : Tensor
            Residual whose last dimension is ``output_size``.

        Returns
        -------
        Tensor
            The normalized sum, whose last dimension is ``output_size``.
        """
        gated: Tensor = functional.glu(self.gate(self.dropout(values)), dim=-1)
        outputs: Tensor = self.normalization(gated + residual)
        return outputs

    def __call__(self, values: Tensor, residual: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values, residual))
