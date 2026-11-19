# anomaly_detection

## Overview

Reconstruction-based anomaly detection with any model that builds a reconstruction network: trained on normal
behavior, it scores every time step by its reconstruction loss.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `AnomalyDetectorSettings`: model, window, threshold quantile, loss, training and scaling. |
| [detector.py](./detector.py) | `AnomalyDetector`: `fit`, `score`, `detect`, `evaluate`, `save` and `load`. |
| [reconstruction_step.py](./reconstruction_step.py) | `ReconstructionStep`: reconstruction loss of a batch and of each step. |
| [metrics.py](./metrics.py) | `AnomalyDetectionMetrics`: precision, recall, F1 and ROC AUC, with point-adjusted variants. |
