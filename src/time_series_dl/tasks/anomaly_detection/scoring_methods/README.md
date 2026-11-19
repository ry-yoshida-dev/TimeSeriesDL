# scoring_methods

## Overview

How the anomaly detector trains a model and scores every time step. Each method builds an `AnomalyScoringNetwork`:
the models' own anomaly scoring networks are used as they are, and reconstruction and forecast networks are wrapped
in adapters that train on and score their errors.

## Components

| Component | Description |
| --------- | ----------- |
| [anomaly_scoring.py](./anomaly_scoring.py) | `AnomalyScoring`: which models a method supports, its window length and the step it builds. |
| [reconstruction_error/](./reconstruction_error/README.md) | `ReconstructionErrorScoring`: reconstruction loss of each step. |
| [forecast_error/](./forecast_error/README.md) | `ForecastErrorScoring`: forecast loss of each step, from any forecasting model. |
| [native/](./native/README.md) | `NativeScoring`: the model's own objective and anomaly score. |
