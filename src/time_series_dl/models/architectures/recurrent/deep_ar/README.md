# deep_ar

## Overview

DeepAR (Salinas et al., "DeepAR: Probabilistic forecasting with autoregressive recurrent networks", International
Journal of Forecasting, 2020): recurrent layers encode the past window and decode the horizon autoregressively, with
teacher forcing while training. With covariates, each step's input also holds its known covariates and the static
features of the series, with embedded categorical codes, as in the paper. Predictive quantiles come from the
objective's distribution instead of sampled paths.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `DeepARParameters`: cell, hidden size, layers and dropout. |
| [recurrence.py](./recurrence.py) | `DeepARRecurrence`: encodes the past and decodes the horizon step by step from the previous targets and each step's features. |
| [forecast_network.py](./forecast_network.py) | `DeepARForecastNetwork`: the recurrence over the input channels. |
| [covariate_forecast_network.py](./covariate_forecast_network.py) | `DeepARCovariateForecastNetwork`: the recurrence with known covariates and static features at every step. |
