from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray


class CalendarFeature(Enum):
    """
    Position of a time stamp within a calendar period, encoded in [-0.5, 0.5] as in GluonTS.

    Attributes
    ----------
    MINUTE_OF_HOUR
        Minute within the hour.
    HOUR_OF_DAY
        Hour within the day.
    DAY_OF_WEEK
        Day within the week, from Monday.
    DAY_OF_MONTH
        Day within the month.
    DAY_OF_YEAR
        Day within the year.
    MONTH_OF_YEAR
        Month within the year.
    """

    MINUTE_OF_HOUR = "minute_of_hour"
    HOUR_OF_DAY = "hour_of_day"
    DAY_OF_WEEK = "day_of_week"
    DAY_OF_MONTH = "day_of_month"
    DAY_OF_YEAR = "day_of_year"
    MONTH_OF_YEAR = "month_of_year"

    def encode(self, timestamps: NDArray[np.datetime64]) -> FloatArray:
        """
        Encode time stamps.

        Parameters
        ----------
        timestamps : NDArray[np.datetime64]
            Time stamps of any shape.

        Returns
        -------
        FloatArray
            The encoded positions in [-0.5, 0.5], with the shape of the time stamps.
        """
        days: NDArray[np.datetime64] = timestamps.astype("datetime64[D]")
        position: NDArray[np.int64]
        last: int
        match self:
            case CalendarFeature.MINUTE_OF_HOUR:
                position = (timestamps.astype("datetime64[m]") - timestamps.astype("datetime64[h]")).astype(np.int64)
                last = 59
            case CalendarFeature.HOUR_OF_DAY:
                position = (timestamps.astype("datetime64[h]") - days).astype(np.int64)
                last = 23
            case CalendarFeature.DAY_OF_WEEK:
                # 1970-01-01, day zero, was a Thursday.
                position = (days.astype(np.int64) + 3) % 7
                last = 6
            case CalendarFeature.DAY_OF_MONTH:
                position = (days - timestamps.astype("datetime64[M]")).astype(np.int64)
                last = 30
            case CalendarFeature.DAY_OF_YEAR:
                position = (days - timestamps.astype("datetime64[Y]")).astype(np.int64)
                last = 365
            case CalendarFeature.MONTH_OF_YEAR:
                position = (timestamps.astype("datetime64[M]") - timestamps.astype("datetime64[Y]")).astype(np.int64)
                last = 11
        return (position / last - 0.5).astype(np.float32)
