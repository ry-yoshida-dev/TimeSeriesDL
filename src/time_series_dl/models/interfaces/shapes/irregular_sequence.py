from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class IrregularSequenceShape:
    """
    Dimensions of irregularly sampled sequences that an interpolation network predicts at query times.

    Attributes
    ----------
    channels : int
        Number of channels.

    Raises
    ------
    ValueError
        If the number of channels is not positive.
    """

    channels: int

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
