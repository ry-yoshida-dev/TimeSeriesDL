# objectives

## Overview

Implementations of `ForecastObjective`: what a forecast network predicts for each future value.

## Components

| Component | Description |
| --------- | ----------- |
| [point.py](./point.py) | `PointObjective`: one value, trained with any `PointLoss`. |
| [quantile.py](./quantile.py) | `QuantileObjective`: non-crossing quantiles, trained with the pinball loss. |
| [gaussian.py](./gaussian.py) | `GaussianObjective`: mean and standard deviation, trained with the negative log-likelihood. |
| [student_t.py](./student_t.py) | `StudentTObjective`: location, scale and degrees of freedom; heavy-tailed likelihood. |
| [laplace.py](./laplace.py) | `LaplaceObjective`: location and scale; likelihood that estimates the median. |
