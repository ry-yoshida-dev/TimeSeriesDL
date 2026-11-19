# covariates

## Overview

Building blocks that give forecast networks known-future and static covariates: the embedding of static features
shared by the models that read covariates natively, and the generic adapter through which every other forecasting
model accepts covariates.

## Components

| Component | Description |
| --------- | ----------- |
| [static_embedding.py](./static_embedding.py) | `StaticCovariateEmbedding`: numeric static features and embedded categorical codes as one vector. |
| [adapter_parameters.py](./adapter_parameters.py) | `CovariateAdapterParameters`: size of the adapter's correction, and the builder of the adapter around any `ForecastingModel`. |
| [adapter_network.py](./adapter_network.py) | `CovariateAdapterNetwork`: past known covariates as extra input channels, and a per-step correction of the outputs from the future known covariates and static features. |
