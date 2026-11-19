from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class IrregularSummaryShape:
    """
    Dimensions of an irregular summary network, which maps a whole irregularly sampled sequence to one output vector.

    Attributes
    ----------
    input_channels : int
        Number of channels of the input.
    output_size : int
        Size of the output vector (number of classes, or of regression targets).

    Raises
    ------
    ValueError
        If a dimension is not positive.
    """

    input_channels: int
    output_size: int

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")
        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}")
