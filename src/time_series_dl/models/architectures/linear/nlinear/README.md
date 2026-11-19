# nlinear

## Overview

NLinear (Zeng et al., AAAI 2023): a linear map along time of the window relative to its last value.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `NLinearParameters`: per-channel weights. |
| [forecast_network.py](./forecast_network.py) | `NLinearForecastNetwork`: maps the shifted window to the horizon. |
