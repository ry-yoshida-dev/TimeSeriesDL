from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SequenceShape:
    """
    Dimensions of a sequence that a reconstruction network maps to a sequence of the same shape.

    Attributes
    ----------
    length : int
        Number of time steps.
    channels : int
        Number of channels.

    Raises
    ------
    ValueError
        If a dimension is not positive.
    """

    length: int
    channels: int

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
