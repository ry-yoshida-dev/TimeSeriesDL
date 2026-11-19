# differential_equations

## Overview

Fixed-step solvers of ordinary differential equations and the cubic control paths of controlled differential
equations, for the continuous-time models. Everything is batched over samples with their own times, and padding
steps that repeat a sample's last time or knot leave the results unchanged.

## Components

| Component | Description |
| --------- | ----------- |
| [vector_field.py](./vector_field.py) | `VectorField`: time and state to the derivative of the state. |
| [butcher_tableau.py](./butcher_tableau.py) | `ButcherTableau`: coefficients of an explicit Runge-Kutta method. |
| [ode_solver.py](./ode_solver.py) | `ODESolver`: fixed-step explicit Runge-Kutta integration between per-sample times. |
| [ode_solver_kind.py](./ode_solver_kind.py) | `ODESolverKind`: Euler, classical fourth-order Runge-Kutta and fixed-step Dormand-Prince. |
| [cubic_path.py](./cubic_path.py) | `CubicPath`: piecewise cubic path, its values and its derivative within an interval. |
| [cubic_interpolation.py](./cubic_interpolation.py) | `CubicInterpolation`: knots to a `CubicPath`. |
| [cubic_interpolations/](./cubic_interpolations/README.md) | Natural cubic spline and Hermite cubic with backward differences. |
| [cubic_interpolation_kind.py](./cubic_interpolation_kind.py) | `CubicInterpolationKind`: choice of the interpolation. |
| [interval_differences.py](./interval_differences.py) | `IntervalDifferences`: widths, differences and slopes between knots. |

## Examples

```python
import torch

from time_series_dl.models.components.differential_equations import ODESolverKind

solver = ODESolverKind.RUNGE_KUTTA.build()
decayed = solver.integrate(lambda time, state: -state, torch.ones(4, 1), torch.zeros(4, 1), torch.ones(4, 1), 20)
```
