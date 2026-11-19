"""Forecast shapes that every forecast-only model is checked against."""

from time_series_dl.models import ForecastShape

FORECAST_SHAPE_CASES: list[ForecastShape] = [
    ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
    ForecastShape(
        input_length=17,
        input_channels=4,
        horizon=5,
        target_channels=(2, 0),
        num_parameters=3,
        location_index=1,
        level_parameters=(0, 1, 2),
    ),
    ForecastShape(input_length=15, input_channels=3, horizon=7, target_channels=(2, 1), num_parameters=2),
    ForecastShape(input_length=16, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
    ForecastShape(input_length=3, input_channels=1, horizon=9, target_channels=(0,)),
]
