import math

import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline
from torch import Tensor

from time_series_dl.models.components.differential_equations import (
    ButcherTableau,
    CubicInterpolationKind,
    CubicPath,
    HermiteCubicInterpolation,
    NaturalCubicInterpolation,
    ODESolverKind,
)

SOLVER_ORDERS: list[tuple[ODESolverKind, int]] = [
    (ODESolverKind.EULER, 1),
    (ODESolverKind.RUNGE_KUTTA, 4),
    (ODESolverKind.DORMAND_PRINCE, 5),
]


def oscillate(time: Tensor, state: Tensor) -> Tensor:
    return torch.stack([state[:, 1], -state[:, 0]], dim=1)


def solve_oscillator(kind: ODESolverKind, num_steps: int) -> float:
    start: Tensor = torch.zeros(1, 1, dtype=torch.float64)
    end: Tensor = torch.full((1, 1), 2.0, dtype=torch.float64)
    state: Tensor = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    solved: Tensor = kind.build().integrate(oscillate, state, start, end, num_steps)
    exact: Tensor = torch.tensor([[math.cos(2.0), -math.sin(2.0)]], dtype=torch.float64)
    return float((solved - exact).abs().max())


@pytest.mark.parametrize(("kind", "order"), SOLVER_ORDERS)
def test_solver_converges_with_its_order_on_the_harmonic_oscillator(kind: ODESolverKind, order: int) -> None:
    coarse: float = solve_oscillator(kind, 8)
    fine: float = solve_oscillator(kind, 16)
    observed_order: float = math.log2(coarse / fine)
    assert abs(observed_order - order) < 0.3
    assert solve_oscillator(kind, 64) < {1: 5e-2, 4: 1e-7, 5: 1e-9}[order]


@pytest.mark.parametrize(("kind", "order"), SOLVER_ORDERS)
def test_solver_matches_exponential_decay_with_per_sample_times(kind: ODESolverKind, order: int) -> None:
    rates: Tensor = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)
    start: Tensor = torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float64)
    end: Tensor = torch.tensor([[1.0], [0.0], [0.5]], dtype=torch.float64)
    initial: Tensor = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    solved: Tensor = kind.build().integrate(lambda _, state: -rates * state, initial, start, end, 200)
    exact: Tensor = initial * torch.exp(-rates * (end - start))
    assert torch.allclose(solved, exact, rtol={1: 5e-2, 4: 1e-10, 5: 1e-10}[order])


@pytest.mark.parametrize("kind", list(ODESolverKind))
def test_zero_length_intervals_leave_the_state_unchanged(kind: ODESolverKind) -> None:
    state: Tensor = torch.randn(4, 3)
    time: Tensor = torch.rand(4, 1)
    solved: Tensor = kind.build().integrate(lambda _, value: value.square() + 1.0, state, time, time, 3)
    assert torch.equal(solved, state)


def test_integrate_through_returns_the_state_at_every_time() -> None:
    solver = ODESolverKind.RUNGE_KUTTA.build()
    times: Tensor = torch.tensor([[0.5, 1.0, 1.0, 2.0], [0.2, 0.4, 0.8, 0.8]], dtype=torch.float64)
    states: Tensor = solver.integrate_through(
        lambda _, state: -state,
        torch.ones(2, 1, dtype=torch.float64),
        torch.zeros(2, 1, dtype=torch.float64),
        times,
        20,
    )
    assert states.shape == (2, 4, 1)
    assert torch.allclose(states[..., 0], torch.exp(-times), atol=1e-9)
    with pytest.raises(ValueError, match="steps_per_interval"):
        _ = solver.integrate_through(lambda _, state: state, torch.ones(2, 1), torch.zeros(2, 1), times.float(), 0)


def test_butcher_tableau_is_validated() -> None:
    assert ODESolverKind.DORMAND_PRINCE.tableau.num_stages == 6
    with pytest.raises(ValueError, match="sum to its node"):
        _ = ButcherTableau(nodes=(0.0, 0.5), coefficients=((), (0.4,)), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="weights must sum to 1"):
        _ = ButcherTableau(nodes=(0.0,), coefficients=((),), weights=(0.9,))
    with pytest.raises(ValueError, match="row 1"):
        _ = ButcherTableau(nodes=(0.0, 0.5), coefficients=((), ()), weights=(0.5, 0.5))


def fit_natural(times: np.ndarray, values: np.ndarray, lengths: list[int]) -> CubicPath:
    return NaturalCubicInterpolation().fit(
        torch.from_numpy(times), torch.from_numpy(values), torch.tensor(lengths, dtype=torch.int64)
    )


def test_natural_spline_reproduces_piecewise_cubic_data_exactly() -> None:
    rng = np.random.default_rng(0)
    # Data drawn from a natural cubic spline, so the interpolation must reproduce it between the knots too.
    reference_knots: np.ndarray = np.sort(rng.uniform(0.0, 10.0, 6))
    reference = CubicSpline(reference_knots, rng.normal(size=(6, 2)), bc_type="natural")
    queries: np.ndarray = np.linspace(float(reference_knots[0]), float(reference_knots[-1]), 200)
    path: CubicPath = fit_natural(reference_knots[None], reference(reference_knots)[None], [6])
    assert np.allclose(path.evaluate(torch.from_numpy(queries[None]))[0].numpy(), reference(queries), atol=1e-10)


def test_natural_spline_matches_scipy_and_ignores_padding() -> None:
    rng = np.random.default_rng(1)
    knots: np.ndarray = np.sort(rng.uniform(0.0, 5.0, 8))
    values: np.ndarray = rng.normal(size=(8, 3))
    padded_times: np.ndarray = np.stack([knots, np.concatenate([knots[:5], np.repeat(knots[4], 3)])])
    padded_values: np.ndarray = np.stack([values, np.concatenate([values[:5], np.zeros((3, 3))])])
    path: CubicPath = fit_natural(padded_times, padded_values, [8, 5])
    full_queries: np.ndarray = np.linspace(float(knots[0]), float(knots[-1]), 50)
    short_queries: np.ndarray = np.linspace(float(knots[0]), float(knots[4]), 50)
    evaluated: np.ndarray = path.evaluate(torch.from_numpy(np.stack([full_queries, short_queries]))).numpy()
    assert np.allclose(evaluated[0], CubicSpline(knots, values, bc_type="natural")(full_queries), atol=1e-10)
    assert np.allclose(evaluated[1], CubicSpline(knots[:5], values[:5], bc_type="natural")(short_queries), atol=1e-10)


@pytest.mark.parametrize("kind", list(CubicInterpolationKind))
def test_interpolation_is_exact_on_linear_data_and_passes_the_knots(kind: CubicInterpolationKind) -> None:
    times: Tensor = torch.tensor([[0.0, 0.3, 1.1, 2.0, 2.4]], dtype=torch.float64)
    linear: Tensor = torch.stack([2.0 * times - 1.0, -0.5 * times], dim=-1)
    path: CubicPath = kind.build().fit(times, linear, torch.tensor([5]))
    queries: Tensor = torch.linspace(0.0, 2.4, 40, dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(path.evaluate(queries), torch.stack([2.0 * queries - 1.0, -0.5 * queries], -1), atol=1e-12)
    rng = np.random.default_rng(2)
    values: Tensor = torch.from_numpy(rng.normal(size=(1, 5, 2)))
    knotted: CubicPath = kind.build().fit(times, values, torch.tensor([5]))
    assert torch.allclose(knotted.evaluate(times), values, atol=1e-12)


def test_hermite_path_is_continuously_differentiable_and_causal() -> None:
    rng = np.random.default_rng(3)
    times: Tensor = torch.from_numpy(np.sort(rng.uniform(0.0, 4.0, (1, 6)), axis=1))
    values: Tensor = torch.from_numpy(rng.normal(size=(1, 6, 2)))
    path: CubicPath = HermiteCubicInterpolation().fit(times, values, torch.tensor([6]))
    widths: Tensor = times[0, 1:] - times[0, :-1]
    for interval in range(4):
        end_slope: Tensor = path.derivative(interval, torch.ones(1, 1, dtype=torch.float64)) / widths[interval]
        start_slope: Tensor = (
            path.derivative(interval + 1, torch.zeros(1, 1, dtype=torch.float64)) / widths[interval + 1]
        )
        assert torch.allclose(end_slope, start_slope, atol=1e-12)
    changed: Tensor = values.clone()
    changed[0, 5] += 10.0
    changed_path: CubicPath = HermiteCubicInterpolation().fit(times, changed, torch.tensor([6]))
    queries: Tensor = torch.linspace(float(times[0, 0]), float(times[0, 4]), 30, dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(path.evaluate(queries), changed_path.evaluate(queries), atol=1e-12)


@pytest.mark.parametrize("kind", list(CubicInterpolationKind))
def test_padding_intervals_have_zero_derivative(kind: CubicInterpolationKind) -> None:
    times: Tensor = torch.tensor([[0.0, 1.0, 2.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
    values: Tensor = torch.randn(2, 5, 3)
    path: CubicPath = kind.build().fit(times, values, torch.tensor([3, 1]))
    for interval in range(2, 4):
        assert torch.equal(path.derivative(interval, torch.rand(2, 1))[0], torch.zeros(3))
    for interval in range(4):
        assert torch.equal(path.derivative(interval, torch.rand(2, 1))[1], torch.zeros(3))
    assert torch.allclose(path.evaluate(torch.tensor([[2.0], [0.7]]))[1, 0], values[1, 0])
    with pytest.raises(ValueError, match="lengths"):
        _ = kind.build().fit(times, values, torch.tensor([0, 1]))
