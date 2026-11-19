from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class TiDEResidualBlock(nn.Module):
    """
    Dense residual block of TiDE: a one-hidden-layer ReLU MLP plus a linear skip, optionally layer-normalized.

    Parameters
    ----------
    input_size : int
        Size of the input vectors.
    hidden_size : int
        Size of the hidden layer.
    output_size : int
        Size of the output vectors.
    dropout : float
        Dropout probability on the MLP output.
    is_layer_normalized : bool
        Whether the sum is layer-normalized.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float, is_layer_normalized: bool
    ) -> None:
        super().__init__()
        self.hidden: nn.Linear = nn.Linear(input_size, hidden_size)
        self.output: nn.Linear = nn.Linear(hidden_size, output_size)
        self.skip: nn.Linear = nn.Linear(input_size, output_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.normalization: nn.Module = nn.LayerNorm(output_size) if is_layer_normalized else nn.Identity()

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``input_size``.

        Returns
        -------
        Tensor
            Outputs whose last dimension is ``output_size``.
        """
        mapped: Tensor = self.dropout(self.output(functional.relu(self.hidden(values))))
        skip: Tensor = self.skip(values)
        outputs: Tensor = self.normalization(mapped + skip)
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
