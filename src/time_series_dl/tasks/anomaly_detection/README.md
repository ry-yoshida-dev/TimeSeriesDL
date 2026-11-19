# anomaly_detection

## Overview

Anomaly detection trained on normal behavior that scores every time step, with three scoring methods:

| Scoring method | Models | Score of a step |
| -------------- | ------ | --------------- |
| `ReconstructionErrorScoring` | any `ReconstructionModel` | reconstruction loss |
| `ForecastErrorScoring` | any `ForecastingModel` | one-step-ahead (or horizon) forecast loss |
| `NativeScoring` | any `AnomalyScoringModel` (Anomaly Transformer, DCdetector, TranAD, USAD) | the model's own criterion, after training on its own objective |

Without an explicit `scoring`, a model's own criterion is preferred, then its reconstruction error, then its forecast
error. Each score is averaged over every window that covers the step, and a quantile of the training scores sets the
threshold.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `AnomalyDetectorSettings`: model, window, scoring method, threshold quantile, loss, training and scaling. |
| [detector.py](./detector.py) | `AnomalyDetector`: `fit`, `score`, `detect`, `evaluate`, `save` and `load`. |
| [scoring_step.py](./scoring_step.py) | `AnomalyScoringStep`: an anomaly scoring network's loss of a batch, with the epoch, and its per-step scores. |
| [scoring_methods/](./scoring_methods/README.md) | `AnomalyScoring` and its reconstruction-error, forecast-error and native methods. |
| [metrics.py](./metrics.py) | `AnomalyDetectionMetrics`: step-wise, point-adjusted (for comparison with papers only) and range-based precision, recall and F1, and ROC AUC. |

## Examples

```python
from time_series_dl.models import DLinearParameters
from time_series_dl.tasks.anomaly_detection import AnomalyDetectorSettings, ForecastErrorScoring

detector = AnomalyDetectorSettings(
    model=DLinearParameters(), window_length=96, scoring=ForecastErrorScoring(horizon=1)
).build()
detector.fit(normal_series)
metrics = detector.evaluate(test_series, labels)
```
