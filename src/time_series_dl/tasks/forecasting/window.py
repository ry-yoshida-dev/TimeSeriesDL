from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastWindow:
    """
    Lengths of the past window a forecast reads and of the horizon it predicts.

    Attributes
    ----------
    input_length : int
        Number of past time steps.
    horizon : int
        Number of future time steps.
    stride : int
        Steps between consecutive training and evaluation windows of a series.

    Raises
    ------
    ValueError
        If a length is not positive.
    """

    input_length: int
    horizon: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.input_length <= 0:
            raise ValueError(f"input_length must be positive, got {self.input_length}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

    @property
    def total_length(self) -> int:
        """
        Get the number of steps of one past window with its horizon.

        Returns
        -------
        int
            The sum of the input length and the horizon.
        """
        return self.input_length + self.horizon
