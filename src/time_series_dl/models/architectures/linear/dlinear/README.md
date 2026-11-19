# dlinear

## Overview

DLinear (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023): linear maps along time
of the moving-average trend and of the seasonal remainder.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `DLinearParameters`: moving-average length and per-channel weights. |
| [decomposition_linear.py](./decomposition_linear.py) | `DecompositionLinear`: the decomposition and its two linear maps. |
| [forecast_network.py](./forecast_network.py) | `DLinearForecastNetwork`: maps the window to the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `DLinearReconstructionNetwork`: maps the window onto itself. |
| [summary_network.py](./summary_network.py) | `DLinearSummaryNetwork`: maps the window, then flattens it to the output vector. |
