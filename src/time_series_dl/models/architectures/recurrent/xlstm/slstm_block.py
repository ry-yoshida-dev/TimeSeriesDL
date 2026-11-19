from typing import cast

from torch import Tensor, nn
from torch.nn import functional

from ....components.state_space import CausalDepthwiseConvolution
from .gated_feedforward import GatedFeedForward
from .headwise_layer_norm import HeadwiseLayerNorm
from .headwise_linear import HeadwiseLinear
from .slstm_cell import SLSTMCell


class SLSTMBlock(nn.Module):
    """
    Residual sLSTM block with post up-projection (Beck et al., 2024).

    The normalized input feeds the cell input and output gates directly and, after a causal convolution and SiLU, the
    input and forget gates, each through a block-diagonal map per head. The cell's output is normalized per head and
    added to the input; a gated feedforward network on the normalized result is added in turn.

    Parameters
    ----------
    size : int
        Size of the input, output and hidden vectors.
    num_heads : int
        Number of heads; it must divide size.
    convolution_size : int
        Kernel size of the causal convolution.
    feedforward_factor : float
        Ratio of the feedforward network's hidden size to size, before rounding up to a multiple of 64.
    dropout : float
        Dropout probability of the cell's output and of the feedforward network's output.
    block_index : int
        Position of the block in the stack.
    num_blocks : int
        Number of blocks of the stack.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide size, or dropout is not in [0, 1).
    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        convolution_size: int,
        feedforward_factor: float,
        dropout: float,
        block_index: int,
        num_blocks: int,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.normalization: nn.LayerNorm = nn.LayerNorm(size, bias=False)
        self.convolution: CausalDepthwiseConvolution = CausalDepthwiseConvolution(size, convolution_size)
        self.input_gate: HeadwiseLinear = HeadwiseLinear(size, size, num_heads, False)
        self.forget_gate: HeadwiseLinear = HeadwiseLinear(size, size, num_heads, False)
        self.cell_input: HeadwiseLinear = HeadwiseLinear(size, size, num_heads, False)
        self.output_gate: HeadwiseLinear = HeadwiseLinear(size, size, num_heads, False)
        self.cell: SLSTMCell = SLSTMCell(size, num_heads, block_index, num_blocks)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.output_normalization: HeadwiseLayerNorm = HeadwiseLayerNorm(size, num_heads)
        self.feedforward_normalization: nn.LayerNorm = nn.LayerNorm(size, bias=False)
        self.feedforward: GatedFeedForward = GatedFeedForward(size, feedforward_factor, dropout)

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
        normalized: Tensor = self.normalization(values)
        convolved: Tensor = functional.silu(self.convolution(normalized))
        hidden: Tensor = self.cell(
            self.input_gate(convolved),
            self.forget_gate(convolved),
            self.cell_input(normalized),
            self.output_gate(normalized),
        )
        mixed: Tensor = values + self.output_normalization(self.dropout(hidden))
        outputs: Tensor = mixed + self.feedforward(self.feedforward_normalization(mixed))
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
