# interfaces

## Overview

Contracts between models and tasks: the abstract networks that models implement, the builder protocols that tasks
require of model parameters, and the dimensions passed to the builders.

## Components

| Component | Description |
| --------- | ----------- |
| [forecast_network.py](./forecast_network.py) | `ForecastNetwork`: past window to the distribution parameters of the horizon. |
| [covariate_forecast_network.py](./covariate_forecast_network.py) | `CovariateForecastNetwork`: past window and its known and static covariates to the distribution parameters of the horizon. |
| [covariate_tensors.py](./covariate_tensors.py) | `CovariateTensors`: known, static numeric and static categorical covariates of a batch. |
| [reconstruction_network.py](./reconstruction_network.py) | `ReconstructionNetwork`: sequence to a sequence of the same shape. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `MaskedReconstructionNetwork`: sequence and observed mask to a reconstruction and the network's own training loss. |
| [masked_reconstruction.py](./masked_reconstruction.py) | `MaskedReconstruction`: reconstruction and optional training loss of a batch. |
| [pretraining_network.py](./pretraining_network.py) | `PretrainingNetwork`: self-supervised loss around a sequence encoder, and its instance-level representation. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `AnomalyScoringNetwork`: window to per-step anomaly scores, with the model's own training loss. |
| [summary_network.py](./summary_network.py) | `SummaryNetwork`: sequence to one output vector. |
| [sequence_encoder.py](./sequence_encoder.py) | `SequenceEncoder`: sequence to a sequence of feature vectors. |
| [protocols/](./protocols/README.md) | Builder protocols that tasks require of model parameters. |
| [shapes/](./shapes/README.md) | Dimensions passed to the builders. |
