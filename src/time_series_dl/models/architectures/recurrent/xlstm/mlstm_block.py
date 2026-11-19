import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from ....components.state_space import CausalDepthwiseConvolution
from .headwise_linear import HeadwiseLinear
from .mlstm_cell import MLSTMCell


class MLSTMBlock(nn.Module):
    """
    Residual mLSTM block with pre up-projection (Beck et al., 2024).

    The normalized input is projected up to an mLSTM branch and an output gate branch. The mLSTM branch is convolved
    causally and activated to give the queries and keys through block-diagonal maps, while the values read the branch
    before the convolution. The cell's output plus a learned skip of the convolved branch is gated by the SiLU of the
    gate branch, projected down, and added to the input.

    Parameters
    ----------
    size : int
        Size of the input and output vectors.
    expansion : int
        Ratio of the inner size to size.
    num_heads : int
        Number of heads of the cell; it must divide the inner size.
    projection_block_size : int
        Size of the blocks of the query, key and value maps; it must divide the inner size.
    convolution_size : int
        Kernel size of the causal convolution.
    dropout : float
        Dropout probability of the block's output.
    num_blocks : int
        Number of blocks of the stack, which scales the initialization of the down-projection.

    Raises
    ------
    ValueError
        If a size is not positive, a head or block size does not divide the inner size, or dropout is not in [0, 1).
    """

    def __init__(
        self,
        size: int,
        expansion: int,
        num_heads: int,
        projection_block_size: int,
        convolution_size: int,
        dropout: float,
        num_blocks: int,
    ) -> None:
        for name, value in (
            ("size", size),
            ("expansion", expansion),
            ("projection_block_size", projection_block_size),
            ("num_blocks", num_blocks),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        inner_size: int = expansion * size
        if inner_size % projection_block_size != 0:
            raise ValueError(f"projection_block_size {projection_block_size} must divide the inner size {inner_size}")
        super().__init__()
        self.normalization: nn.LayerNorm = nn.LayerNorm(size, bias=False)
        self.up_projection: nn.Linear = nn.Linear(size, 2 * inner_size, bias=False)
        self.convolution: CausalDepthwiseConvolution = CausalDepthwiseConvolution(inner_size, convolution_size)
        num_projection_heads: int = inner_size // projection_block_size
        self.query_projection: HeadwiseLinear = HeadwiseLinear(inner_size, inner_size, num_projection_heads, False)
        self.key_projection: HeadwiseLinear = HeadwiseLinear(inner_size, inner_size, num_projection_heads, False)
        self.value_projection: HeadwiseLinear = HeadwiseLinear(inner_size, inner_size, num_projection_heads, False)
        self.cell: MLSTMCell = MLSTMCell(inner_size, num_heads)
        self.skip: nn.Parameter = nn.Parameter(torch.ones(inner_size))
        self.down_projection: nn.Linear = nn.Linear(inner_size, size, bias=False)
        # Small initialization of the input projection and depth-scaled initialization of the output projection.
        _ = nn.init.normal_(self.up_projection.weight, std=math.sqrt(2.0 / (5 * size)))
        _ = nn.init.normal_(self.down_projection.weight, std=2.0 / num_blocks / math.sqrt(size))
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Features with shape (batch, length, size).

        Returns
        -------
        Tensor
            Features with the shape of ``values``.
        """
        branch, gate = self.up_projection(self.normalization(values)).chunk(2, dim=-1)
        convolved: Tensor = functional.silu(self.convolution(branch))
        hidden: Tensor = self.cell(
            self.query_projection(convolved), self.key_projection(convolved), self.value_projection(branch)
        )
        gated: Tensor = (hidden + self.skip * convolved) * functional.silu(gate)
        outputs: Tensor = values + self.dropout(self.down_projection(gated))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
