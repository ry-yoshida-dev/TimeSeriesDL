# conformal

## Overview

Calibrated forecast intervals around any fitted `Forecaster`: split conformal prediction on held-out calibration
series, with a separate interval for every horizon step and target channel, and adaptive conformal inference
(Gibbs and Candès, 2021) that keeps the coverage under distribution shift by updating the intervals online.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ConformalSettings`: coverages, conformity score and adaptation rate. |
| [conformity_score.py](./conformity_score.py) | `ConformityScore`: absolute residual of the point forecast, or quantile residual (conformalized quantile regression). |
| [forecaster.py](./forecaster.py) | `ConformalForecaster`: `calibrate`, `predict`, `forecast`, online `update` and `evaluate`. |

## Examples

```python
from time_series_dl.tasks.forecasting import ConformalSettings

conformal = ConformalSettings(coverage_levels=(0.8, 0.9)).build(forecaster)  # forecaster is fitted
conformal.calibrate(calibration_series)
interval = conformal.forecast(series)  # interval.get_quantile(0.05), interval.get_quantile(0.95)
conformal.update(past_windows, their_true_futures)  # adaptive conformal inference
```
