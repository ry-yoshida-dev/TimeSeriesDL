from dataclasses import dataclass

from .block_stack import XLSTMBlockStack


@dataclass(frozen=True, kw_only=True)
class XLSTMStackParameters:
    """
    Configuration of a stack of xLSTM blocks, each an mLSTM or an sLSTM block.

    The defaults are those of the official xLSTM example configuration: seven blocks with an sLSTM block at index 1,
    four heads in both kinds of cell, causal convolutions of size 4, query, key and value maps with blocks of 4,
    an mLSTM up-projection of 2, an sLSTM feedforward factor of 1.3 and no dropout.

    Attributes
    ----------
    num_blocks : int
        Number of blocks.
    slstm_block_indices : tuple[int, ...]
        Positions of the sLSTM blocks; every other block is an mLSTM block.
    mlstm_num_heads : int
        Number of heads of each mLSTM cell; it must divide the inner size of the mLSTM blocks.
    slstm_num_heads : int
        Number of heads of each sLSTM cell; it must divide the size of the stack.
    mlstm_expansion : int
        Ratio of the inner size of each mLSTM block to the size of the stack.
    projection_block_size : int
        Size of the blocks of the mLSTM query, key and value maps; it must divide the inner size.
    convolution_size : int
        Kernel size of the causal convolution of each block.
    feedforward_factor : float
        Ratio of the hidden size of the sLSTM feedforward networks to the size of the stack, before rounding up to a
        multiple of 64.
    dropout : float
        Dropout probability of each block's output.

    Raises
    ------
    ValueError
        If a count or size is not positive, an sLSTM index is out of range or repeated, or dropout is not in [0, 1).
    """

    num_blocks: int = 7
    slstm_block_indices: tuple[int, ...] = (1,)
    mlstm_num_heads: int = 4
    slstm_num_heads: int = 4
    mlstm_expansion: int = 2
    projection_block_size: int = 4
    convolution_size: int = 4
    feedforward_factor: float = 1.3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("mlstm_num_heads", self.mlstm_num_heads),
            ("slstm_num_heads", self.slstm_num_heads),
            ("mlstm_expansion", self.mlstm_expansion),
            ("projection_block_size", self.projection_block_size),
            ("convolution_size", self.convolution_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.feedforward_factor <= 0.0:
            raise ValueError(f"feedforward_factor must be positive, got {self.feedforward_factor}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if len(set(self.slstm_block_indices)) != len(self.slstm_block_indices):
            raise ValueError(f"slstm_block_indices must not repeat, got {self.slstm_block_indices}")
        if any(index < 0 or index >= self.num_blocks for index in self.slstm_block_indices):
            raise ValueError(f"slstm_block_indices must be in [0, {self.num_blocks}), got {self.slstm_block_indices}")

    @property
    def is_slstm_used(self) -> bool:
        """
        Check whether the stack has an sLSTM block.

        Returns
        -------
        bool
            True if at least one block is an sLSTM block.
        """
        return bool(self.slstm_block_indices)

    @property
    def is_mlstm_used(self) -> bool:
        """
        Check whether the stack has an mLSTM block.

        Returns
        -------
        bool
            True if at least one block is an mLSTM block.
        """
        return len(self.slstm_block_indices) < self.num_blocks

    def check_size(self, size: int) -> None:
        """
        Check that the heads and blocks divide the sizes of a stack of the given size.

        Parameters
        ----------
        size : int
            Size of the features of the stack.

        Raises
        ------
        ValueError
            If size is not positive or a head count or block size does not divide the size it splits.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if self.is_slstm_used and size % self.slstm_num_heads != 0:
            raise ValueError(f"slstm_num_heads {self.slstm_num_heads} must divide the model size {size}")
        inner_size: int = self.mlstm_expansion * size
        if self.is_mlstm_used and inner_size % self.mlstm_num_heads != 0:
            raise ValueError(f"mlstm_num_heads {self.mlstm_num_heads} must divide the mLSTM inner size {inner_size}")
        if self.is_mlstm_used and inner_size % self.projection_block_size != 0:
            raise ValueError(
                f"projection_block_size {self.projection_block_size} must divide the mLSTM inner size {inner_size}"
            )

    def build(self, size: int) -> XLSTMBlockStack:
        """
        Build the untrained stack.

        Parameters
        ----------
        size : int
            Size of the features of the stack.

        Returns
        -------
        XLSTMBlockStack
            The stack.

        Raises
        ------
        ValueError
            If the heads or blocks do not divide the sizes they split.
        """
        self.check_size(size)
        return XLSTMBlockStack(self, size)
