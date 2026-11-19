from dataclasses import dataclass

from .block_kind import NBEATSBlockKind


@dataclass(frozen=True, kw_only=True)
class NBEATSStack:
    """
    Configuration of one N-BEATS stack: consecutive blocks with the same basis.

    Attributes
    ----------
    kind : NBEATSBlockKind
        Basis of every block of the stack.
    num_blocks : int
        Number of blocks.
    num_layers : int
        Number of fully connected hidden layers of each block.
    hidden_size : int
        Width of the hidden layers.
    is_weight_shared : bool
        Whether every block of the stack reuses the weights of the first, as in the interpretable architecture.

    Raises
    ------
    ValueError
        If a count or size is not positive.
    """

    kind: NBEATSBlockKind = NBEATSBlockKind.GENERIC
    num_blocks: int = 1
    num_layers: int = 4
    hidden_size: int = 512
    is_weight_shared: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
