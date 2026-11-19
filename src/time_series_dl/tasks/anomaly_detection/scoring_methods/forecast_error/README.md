# forecast_error

## Overview

Forecast-error scoring: any forecasting model predicts the next steps of every channel from the window before them,
and each step is scored by its forecast loss averaged over channels and over the forecasts that cover it.

## Components

| Component | Description |
| --------- | ----------- |
| [anomaly_scoring.py](./anomaly_scoring.py) | `ForecastErrorScoring`: horizon and instance normalization; builds the forecast network and its adapter. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `ForecastErrorScoringNetwork`: forecast loss for training and for the scores. |
