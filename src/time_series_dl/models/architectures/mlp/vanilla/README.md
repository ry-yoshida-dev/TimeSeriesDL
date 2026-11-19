# vanilla

## Overview

Multilayer perceptron over the flattened window: a baseline for every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MLPParameters`: hidden sizes, activation and dropout. |
| [forecast_network.py](./forecast_network.py) | `MLPForecastNetwork`: flattened window to flattened horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `MLPReconstructionNetwork`: flattened window to its reconstruction. |
| [summary_network.py](./summary_network.py) | `MLPSummaryNetwork`: flattened window to the output vector. |
