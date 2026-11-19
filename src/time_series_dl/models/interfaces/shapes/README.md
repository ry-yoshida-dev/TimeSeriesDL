# shapes

## Overview

Input and output dimensions that a task passes to a model when it builds a network.

## Components

| Component | Description |
| --------- | ----------- |
| [forecast.py](./forecast.py) | `ForecastShape`: input window, horizon, target channels and distribution parameters of a forecast network. |
| [covariate_forecast.py](./covariate_forecast.py) | `CovariateForecastShape`: a `ForecastShape` with the known, static numeric and static categorical covariate dimensions. |
| [sequence.py](./sequence.py) | `SequenceShape`: length and channels of a reconstruction network's input and output. |
| [summary.py](./summary.py) | `SummaryShape`: input window and output size of a summary network. |
