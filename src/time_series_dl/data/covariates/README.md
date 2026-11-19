# covariates

## Overview

Validation, windowing and scaling of the covariates given to a forecaster next to its series: known covariates that
cover the past and the horizon, and static numeric and categorical features of each series.

## Components

| Component | Description |
| --------- | ----------- |
| [series_covariates.py](./series_covariates.py) | `SeriesCovariates`: known, static numeric and static categorical covariates of one series. |
| [window_covariates.py](./window_covariates.py) | `WindowCovariates`: covariates of a batch of forecast windows. |
| [collection.py](./collection.py) | `CovariateCollection`: covariates of every series, aligned with a `SeriesCollection` and cut into windows. |
| [scaler.py](./scaler.py) | `CovariateScaler`: scaling fitted on the training covariates, and the categories seen in training. |
| [dimensions.py](./dimensions.py) | `CovariateDimensions`: number of covariates of each kind. |
| [mismatch_error.py](./mismatch_error.py) | `CovariateMismatchError`: covariates that do not fit the series or the fitted forecaster. |
| [array_validation.py](./array_validation.py) | `CovariateArrayValidation`: checks of covariate arrays and categorical codes. |
| [calendar_feature.py](./calendar_feature.py) | `CalendarFeature`: position of a time stamp within an hour, day, week, month or year. |
| [calendar_features.py](./calendar_features.py) | `CalendarFeatures`: known covariates computed from time stamps. |

## Examples

```python
import numpy as np

from time_series_dl.data import CalendarFeatures, SeriesCovariates

timestamps = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[h]")
covariates = SeriesCovariates(
    known=CalendarFeatures().compute(timestamps),  # (length, 4)
    static=np.array([12.5]),
    static_categories=np.array([3]),  # e.g. store id 3
)
```
