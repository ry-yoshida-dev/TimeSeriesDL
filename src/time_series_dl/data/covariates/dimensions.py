from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class CovariateDimensions:
    """
    Number of covariates of each kind.

    Attributes
    ----------
    known_channels : int
        Number of channels of the known covariates.
    static_channels : int
        Number of numeric static features.
    static_categories : int
        Number of categorical static features.

    Raises
    ------
    ValueError
        If a count is negative.
    """

    known_channels: int = 0
    static_channels: int = 0
    static_categories: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("known_channels", self.known_channels),
            ("static_channels", self.static_channels),
            ("static_categories", self.static_categories),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
