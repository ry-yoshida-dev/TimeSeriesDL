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
| [summary.py](./summary.py) | `SummaryModel`: builds a `SummaryNetwork`. |
