# deep_ar

## Overview

DeepAR (Salinas et al., "DeepAR: Probabilistic forecasting with autoregressive recurrent networks", International
Journal of Forecasting, 2020): recurrent layers encode the past window and decode the horizon autoregressively, with
teacher forcing while training. Predictive quantiles come from the objective's distribution instead of sampled paths.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `DeepARParameters`: cell, hidden size, layers and dropout. |
| [forecast_network.py](./forecast_network.py) | `DeepARForecastNetwork`: encodes the past and decodes the horizon step by step. |
