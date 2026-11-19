from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SummaryShape:
    """
    Dimensions of a summary network, which maps a whole sequence to one output vector.

    Attributes
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    output_size : int
        Size of the output vector (number of classes, or of regression targets).

    Raises
    ------
    ValueError
        If a dimension is not positive.
    """

    input_length: int
    input_channels: int
    output_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("input_length", self.input_length),
            ("input_channels", self.input_channels),
            ("output_size", self.output_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
