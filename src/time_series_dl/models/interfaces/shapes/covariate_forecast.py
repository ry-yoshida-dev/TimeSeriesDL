from dataclasses import dataclass

from .forecast import ForecastShape


@dataclass(frozen=True, kw_only=True)
class CovariateForecastShape:
    """
    Input and output dimensions of a forecast network that also reads covariates.

    Attributes
    ----------
    forecast : ForecastShape
        Dimensions of the past values (including past-observed covariate channels) and of the forecast.
    known_channels : int
        Number of channels of the known covariates, which cover the past window and the horizon.
    static_channels : int
        Number of numeric static features of each series, including any the task adds.
    static_cardinalities : tuple[int, ...]
        Number of categories of each categorical static feature, whose values are integer codes in
        ``[0, cardinality)``.

    Raises
    ------
    ValueError
        If a count is negative, a cardinality is not positive, or there is no covariate at all.
    """

    forecast: ForecastShape
    known_channels: int = 0
    static_channels: int = 0
    static_cardinalities: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.known_channels < 0:
            raise ValueError(f"known_channels must not be negative, got {self.known_channels}")
        if self.static_channels < 0:
            raise ValueError(f"static_channels must not be negative, got {self.static_channels}")
        if any(cardinality <= 0 for cardinality in self.static_cardinalities):
            raise ValueError(f"static_cardinalities must be positive, got {self.static_cardinalities}")
        if not (self.is_known_present or self.is_static_present):
            raise ValueError("a covariate forecast needs known or static covariates.")

    @property
    def static_categories(self) -> int:
        """
        Get the number of categorical static features.

        Returns
        -------
        int
            The number of features.
        """
        return len(self.static_cardinalities)

    @property
    def is_known_present(self) -> bool:
        """
        Check whether there are known covariates.

        Returns
        -------
        bool
            True if ``known_channels`` is positive.
        """
        return self.known_channels > 0

    @property
    def is_static_present(self) -> bool:
        """
        Check whether there are static features.

        Returns
        -------
        bool
            True if there is a numeric or a categorical static feature.
        """
        return self.static_channels > 0 or self.static_categories > 0
