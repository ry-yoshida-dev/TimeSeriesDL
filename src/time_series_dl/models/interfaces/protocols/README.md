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
| [irregular_summary.py](./irregular_summary.py) | `IrregularSummaryModel`: builds an `IrregularSummaryNetwork`. |
| [irregular_interpolation.py](./irregular_interpolation.py) | `IrregularInterpolationModel`: builds an `IrregularInterpolationNetwork`. |
| [generative_forecasting.py](./generative_forecasting.py) | `GenerativeForecastingModel`: builds a `GenerativeForecastNetwork`. |
| [published_weights.py](./published_weights.py) | `PublishedWeightsModel`: names published pretrained weights and can drop them before a checkpoint is restored. |
| [zero_shot_forecasting.py](./zero_shot_forecasting.py) | `ZeroShotForecastingModel`: forecasts with its published weights without training. |
| [summary.py](./summary.py) | `SummaryModel`: builds a `SummaryNetwork`. |
