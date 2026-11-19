# cubic_interpolations

## Overview

Implementations of `CubicInterpolation`, the control paths of neural controlled differential equations.

## Components

| Component | Description |
| --------- | ----------- |
| [natural.py](./natural.py) | `NaturalCubicInterpolation`: natural cubic spline, twice continuously differentiable. |
| [hermite.py](./hermite.py) | `HermiteCubicInterpolation`: Hermite cubic with backward differences, causal. |
