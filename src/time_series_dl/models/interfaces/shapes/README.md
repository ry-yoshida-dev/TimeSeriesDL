# shapes

## Overview

Input and output dimensions that a task passes to a model when it builds a network.

## Components

| Component | Description |
| --------- | ----------- |
| [forecast.py](./forecast.py) | `ForecastShape`: input window, horizon, target channels and distribution parameters of a forecast network. |
| [sequence.py](./sequence.py) | `SequenceShape`: length and channels of a reconstruction network's input and output. |
| [summary.py](./summary.py) | `SummaryShape`: input window and output size of a summary network. |
