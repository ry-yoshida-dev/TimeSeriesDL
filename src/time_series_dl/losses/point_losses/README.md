# point_losses

## Overview

Implementations of `PointLoss`.

## Components

| Component | Description |
| --------- | ----------- |
| [mse.py](./mse.py) | `MSELoss`: squared error; estimates the mean. |
| [mae.py](./mae.py) | `MAELoss`: absolute error; estimates the median. |
| [huber.py](./huber.py) | `HuberLoss`: quadratic up to `delta`, linear beyond. |
| [log_cosh.py](./log_cosh.py) | `LogCoshLoss`: smooth loss between squared and absolute error. |
| [pinball.py](./pinball.py) | `PinballLoss`: quantile loss of one level. |
