from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ModernTCNStage:
    """
    One stage of ModernTCN: blocks at one resolution and feature size.

    Attributes
    ----------
    num_blocks : int
        Number of ModernTCN blocks.
    large_kernel_size : int
        Length of the large depthwise kernel; odd, so the length is kept.
    small_kernel_size : int
        Length of the small depthwise kernel added to the large one; odd and at most large_kernel_size.
    model_size : int
        Number of features per variable and patch.

    Raises
    ------
    ValueError
        If a size is not positive, a kernel length is even, or the small kernel is longer than the large one.
    """

    num_blocks: int = 1
    large_kernel_size: int = 51
    small_kernel_size: int = 5
    model_size: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("large_kernel_size", self.large_kernel_size),
            ("small_kernel_size", self.small_kernel_size),
            ("model_size", self.model_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.large_kernel_size % 2 == 0 or self.small_kernel_size % 2 == 0:
            raise ValueError(f"kernel sizes must be odd, got {self.large_kernel_size} and {self.small_kernel_size}")
        if self.small_kernel_size > self.large_kernel_size:
            raise ValueError(
                f"small_kernel_size {self.small_kernel_size} exceeds large_kernel_size {self.large_kernel_size}"
            )
