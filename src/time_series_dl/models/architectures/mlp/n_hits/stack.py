from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class NHITSStack:
    """
    Configuration of one N-HiTS stack: blocks that read the window at one rate and forecast at one resolution.

    Attributes
    ----------
    pooling_kernel_size : int
        Length of the pooling windows of the input; longer windows make the stack focus on low frequencies.
    downsampling_factor : int
        Number of forecast steps per coefficient, the inverse of the paper's expressiveness ratio; the block
        predicts ceil(horizon / downsampling_factor) coefficients that are interpolated to the horizon.
    num_blocks : int
        Number of blocks.
    num_layers : int
        Number of fully connected hidden layers of each block.
    hidden_size : int
        Width of the hidden layers.

    Raises
    ------
    ValueError
        If a count or size is not positive.
    """

    pooling_kernel_size: int
    downsampling_factor: int
    num_blocks: int = 1
    num_layers: int = 2
    hidden_size: int = 512

    def __post_init__(self) -> None:
        for name, value in (
            ("pooling_kernel_size", self.pooling_kernel_size),
            ("downsampling_factor", self.downsampling_factor),
            ("num_blocks", self.num_blocks),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
