from enum import Enum


class TimesFMFrequency(Enum):
    """
    Frequency category of the series, whose learned embedding TimesFM adds to every patch.

    Attributes
    ----------
    HIGH
        Up to daily granularity (embedding 0), the official default.
    MEDIUM
        Weekly and monthly granularity (embedding 1).
    LOW
        Quarterly and coarser granularity (embedding 2).
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def index(self) -> int:
        """
        Get the row of the frequency embedding.

        Returns
        -------
        int
            The index.
        """
        match self:
            case TimesFMFrequency.HIGH:
                return 0
            case TimesFMFrequency.MEDIUM:
                return 1
            case TimesFMFrequency.LOW:
                return 2
