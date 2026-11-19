# protocols

## Overview

Structural types that tasks require of model parameters: one builder method per kind of network. A model supports a
task exactly when it has the builder, so the type checker rejects unsupported combinations.

## Components

| Component | Description |
| --------- | ----------- |
| [forecasting.py](./forecasting.py) | `ForecastingModel`: builds a `ForecastNetwork`. |
| [covariate_forecasting.py](./covariate_forecasting.py) | `CovariateForecastingModel`: also builds a `CovariateForecastNetwork` that reads covariates natively. |
| [reconstruction.py](./reconstruction.py) | `ReconstructionModel`: builds a `ReconstructionNetwork`. |
| [masked_reconstruction.py](./masked_reconstruction.py) | `MaskedReconstructionModel`: builds a `MaskedReconstructionNetwork`, for imputation. |
| [pretraining.py](./pretraining.py) | `PretrainingModel`: builds a `PretrainingNetwork`, for representation learning. |
| [encoding.py](./encoding.py) | `EncodingModel`: builds a `SequenceEncoder` that pretraining methods wrap. |
| [anomaly_scoring.py](./anomaly_scoring.py) | `AnomalyScoringModel`: builds an `AnomalyScoringNetwork`, for anomaly detection. |
| [summary.py](./summary.py) | `SummaryModel`: builds a `SummaryNetwork`. |
