from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .mlstm_block import MLSTMBlock
from .slstm_block import SLSTMBlock

if TYPE_CHECKING:
    from .stack_parameters import XLSTMStackParameters


class XLSTMBlockStack(nn.Module):
    """
    Residual mLSTM and sLSTM blocks in the configured order, followed by a layer normalization.

    Parameters
    ----------
    parameters : XLSTMStackParameters
        The configuration of the blocks.
    size : int
        Size of the features.
    """

    def __init__(self, parameters: XLSTMStackParameters, size: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            SLSTMBlock(
                size,
                parameters.slstm_num_heads,
                parameters.convolution_size,
                parameters.feedforward_factor,
                parameters.dropout,
                index,
                parameters.num_blocks,
            )
            if index in parameters.slstm_block_indices
            else MLSTMBlock(
                size,
                parameters.mlstm_expansion,
                parameters.mlstm_num_heads,
                parameters.projection_block_size,
                parameters.convolution_size,
                parameters.dropout,
                parameters.num_blocks,
            )
            for index in range(parameters.num_blocks)
        ]
        self.blocks: nn.Sequential = nn.Sequential(*blocks)
        self.normalization: nn.LayerNorm = nn.LayerNorm(size, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the blocks.

        Parameters
        ----------
        values : Tensor
            Features with shape (batch, length, size).

        Returns
        -------
        Tensor
            Features with the shape of ``values``.
        """
        outputs: Tensor = self.normalization(self.blocks(values))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
