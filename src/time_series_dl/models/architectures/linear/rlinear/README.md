# rlinear

## Overview

RLinear (Li et al., "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping", 2023): one
linear map along time, wrapped in reversible instance normalization (RevIN) with a learnable affine transformation.

The forecasting task already normalizes each window by its statistics (RevIN without the affine transformation);
RLinear keeps its own affine RevIN, which then reduces to the affine part, and also normalizes when the task's
normalization is turned off. Each target channel is mapped on its own, so channels that are not forecast are not read.
Only the parameters in the scale of the values are denormalized; the others stay in the normalized scale.
Reconstruction maps the window onto itself with the same layers.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `RLinearParameters`: per-channel weights, dropout and the affine transformation. |
| [forecast_network.py](./forecast_network.py) | `RLinearForecastNetwork`: maps each normalized target channel to the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `RLinearReconstructionNetwork`: maps each normalized channel onto itself. |
