import math
from typing import cast

import numpy as np
import pytest
import torch
from torch.distributions import LowRankMultivariateNormal

from time_series_dl import HuberLoss, LogCoshLoss, MAELoss, MSELoss, PinballLoss, PointLoss
from time_series_dl.tasks.forecasting import (
    ForecastMetrics,
    ForecastObjective,
    ForecastPrediction,
    GaussianObjective,
    LaplaceObjective,
    LowRankGaussianObjective,
    PointObjective,
    QuantileObjective,
    SampleObjective,
    SamplePointEstimate,
    StudentTObjective,
)


def test_pinball_loss_weights_errors_by_level() -> None:
    objective = QuantileObjective((0.1, 0.9))
    parameters = torch.zeros(1, 1, 1, 2)
    targets = torch.ones(1, 1, 1)
    assert torch.isclose(objective.compute_loss(parameters, targets), torch.tensor((0.1 + 0.9) / 2))
    assert torch.isclose(objective.compute_loss(parameters, -targets), torch.tensor((0.9 + 0.1) / 2))


def test_quantile_objective_sorts_and_picks_median() -> None:
    objective = QuantileObjective((0.2, 0.5, 0.8))
    constrained = objective.constrain(torch.tensor([[[[3.0, 1.0, 2.0]]]]))
    assert constrained.tolist() == [[[[1.0, 2.0, 3.0]]]]
    assert objective.location_index == 1
    assert objective.compute_point(constrained).item() == 2.0


def test_gaussian_objective_constrains_and_rescales() -> None:
    objective = GaussianObjective(prediction_quantiles=(0.5,))
    parameters = objective.constrain(torch.tensor([[[[1.0, -50.0]]]]))
    assert parameters[..., 1].item() > 0.0
    rescaled = objective.rescale(torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[10.0]]]), torch.tensor([[[3.0]]]))
    assert rescaled.tolist() == [[[[13.0, 6.0]]]]
    assert objective.compute_quantiles(rescaled).item() == pytest.approx(13.0)


def test_gaussian_loss_is_negative_log_likelihood() -> None:
    objective = GaussianObjective()
    loss = objective.compute_loss(torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[0.0]]]))
    assert loss.item() == pytest.approx(0.5 * math.log(2.0 * math.pi))


@pytest.mark.parametrize("loss", [MSELoss(), MAELoss(), HuberLoss(), LogCoshLoss(), PinballLoss(0.9)])
def test_point_objective_uses_loss(loss: PointLoss) -> None:
    objective = PointObjective(loss)
    assert objective.compute_loss(torch.full((1, 2, 1, 1), 2.0), torch.zeros(1, 2, 1)).item() > 0.0


def test_student_t_objective_keeps_degrees_of_freedom_when_rescaling() -> None:
    objective = StudentTObjective(prediction_quantiles=(0.1, 0.5, 0.9))
    parameters = objective.constrain(torch.tensor([[[[1.0, -50.0, -50.0]]]]))
    assert parameters[..., 1].item() > 0.0
    assert parameters[..., 2].item() >= 2.0
    rescaled = objective.rescale(torch.tensor([[[[1.0, 2.0, 5.0]]]]), torch.tensor([[[10.0]]]), torch.tensor([[[3.0]]]))
    assert rescaled.tolist() == [[[[13.0, 6.0, 5.0]]]]
    quantiles = objective.compute_quantiles(rescaled)
    assert quantiles[..., 1].item() == pytest.approx(13.0)
    # The 0.9 quantile of Student's t with 5 degrees of freedom is 1.475884.
    assert quantiles[..., 2].item() == pytest.approx(13.0 + 6.0 * 1.475884, abs=1e-4)
    assert quantiles[..., 0].item() == pytest.approx(13.0 - 6.0 * 1.475884, abs=1e-4)


def test_student_t_loss_is_negative_log_likelihood() -> None:
    objective = StudentTObjective()
    loss = objective.compute_loss(torch.tensor([[[[0.0, 1.0, 1.0]]]]), torch.tensor([[[0.0]]]))
    assert loss.item() == pytest.approx(math.log(math.pi))


def test_laplace_objective() -> None:
    objective = LaplaceObjective(prediction_quantiles=(0.25, 0.5))
    loss = objective.compute_loss(torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[2.0]]]))
    assert loss.item() == pytest.approx(math.log(2.0) + 2.0)
    rescaled = objective.rescale(torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[10.0]]]), torch.tensor([[[3.0]]]))
    assert rescaled.tolist() == [[[[13.0, 6.0]]]]
    quantiles = objective.compute_quantiles(rescaled)
    assert quantiles[..., 0].item() == pytest.approx(13.0 - 6.0 * math.log(2.0))
    assert quantiles[..., 1].item() == pytest.approx(13.0)


def test_objectives_reject_invalid_levels() -> None:
    with pytest.raises(ValueError, match="increasing"):
        QuantileObjective((0.5, 0.1))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        QuantileObjective((0.0, 0.5))
    with pytest.raises(ValueError, match="empty"):
        QuantileObjective(())
    with pytest.raises(ValueError, match="increasing"):
        StudentTObjective((0.9, 0.1))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        LaplaceObjective((1.0,))


def test_forecast_metrics() -> None:
    prediction = ForecastPrediction(
        point=np.array([[[1.0], [2.0]]]),
        quantile_levels=(0.5,),
        quantiles=np.array([[[[1.0]], [[2.0]]]]),
    )
    metrics = ForecastMetrics.compute(prediction, np.array([[[2.0], [2.0]]]))
    assert metrics.mae == pytest.approx(0.5)
    assert metrics.mse == pytest.approx(0.5)
    assert metrics.mape == pytest.approx(0.25)
    assert metrics.smape == pytest.approx((2.0 / 3.0) / 2.0)
    assert metrics.quantile_loss == pytest.approx(0.25)
    zero_metrics = ForecastMetrics.compute(prediction, np.zeros((1, 2, 1)))
    assert zero_metrics.mape is None


@pytest.mark.parametrize(
    "objective",
    [
        PointObjective(),
        QuantileObjective((0.1, 0.5, 0.9)),
        QuantileObjective((0.5,)),
        GaussianObjective(),
        StudentTObjective(),
        LaplaceObjective(),
        LowRankGaussianObjective(rank=2),
        SampleObjective(num_samples=5),
    ],
)
def test_objective_sampling_shapes_and_seed(objective: ForecastObjective) -> None:
    _ = torch.manual_seed(0)
    parameters = objective.constrain(torch.randn(2, 3, 4, objective.num_parameters))
    first = objective.sample(parameters, 6, torch.Generator().manual_seed(0))
    second = objective.sample(parameters, 6, torch.Generator().manual_seed(0))
    assert first.shape == (2, 6, 3, 4)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(first, second)


@pytest.mark.parametrize("objective", [GaussianObjective(), StudentTObjective(), LaplaceObjective()])
def test_samples_match_the_predicted_quantiles(objective: ForecastObjective) -> None:
    raw = torch.zeros(1, 1, 1, objective.num_parameters)
    raw[..., 0] = 2.0
    parameters = objective.constrain(raw)
    samples = objective.sample(parameters, 40000, torch.Generator().manual_seed(0))
    empirical = torch.quantile(samples.flatten(), torch.tensor(objective.quantile_levels))
    assert torch.allclose(empirical, objective.compute_quantiles(parameters).flatten(), atol=0.05)


def test_quantile_objective_samples_interpolate_between_quantiles() -> None:
    objective = QuantileObjective((0.1, 0.9))
    parameters = torch.tensor([[[[0.0, 8.0]]]])
    samples = objective.sample(parameters, 20000, torch.Generator().manual_seed(0)).flatten()
    assert float(samples.min()) == 0.0 and float(samples.max()) == 8.0
    # Uniform between the quantiles: the median is at 4, and 10% of the mass is clamped to each end.
    assert float(samples.median()) == pytest.approx(4.0, abs=0.1)
    assert float((samples == 0.0).float().mean()) == pytest.approx(0.1, abs=0.01)


def _low_rank_parameters(batch: int, horizon: int, channels: int, rank: int) -> torch.Tensor:
    _ = torch.manual_seed(1)
    return LowRankGaussianObjective(rank=rank).constrain(torch.randn(batch, horizon, channels, rank + 2))


def test_low_rank_gaussian_log_likelihood_matches_torch() -> None:
    objective = LowRankGaussianObjective(rank=3)
    parameters = _low_rank_parameters(2, 4, 5, 3)
    targets = torch.randn(2, 4, 5)
    distribution = LowRankMultivariateNormal(parameters[..., 0], parameters[..., 2:], parameters[..., 1].square())
    reference = cast(torch.Tensor, distribution.log_prob(targets))
    assert torch.allclose(objective.compute_log_likelihood(parameters, targets), reference, atol=1e-4)
    assert objective.compute_loss(parameters, targets).item() == pytest.approx(-float(reference.mean()) / 5, rel=1e-5)


def test_low_rank_gaussian_samples_have_the_covariance() -> None:
    objective = LowRankGaussianObjective(rank=2)
    parameters = _low_rank_parameters(1, 1, 3, 2)
    samples = objective.sample(parameters, 100000, torch.Generator().manual_seed(0))[0, :, 0, :].double()
    factor = parameters[0, 0, :, 2:].double()
    covariance = factor @ factor.T + torch.diag(parameters[0, 0, :, 1].double().square())
    assert torch.allclose(torch.cov(samples.T), covariance, atol=0.05 * float(covariance.abs().max()))
    assert torch.allclose(samples.mean(0), parameters[0, 0, :, 0].double(), atol=0.03)
    marginal = covariance.diagonal().sqrt()
    quantiles = objective.compute_quantiles(parameters)[0, 0]
    assert torch.allclose(quantiles[:, 2] - quantiles[:, 0], (2 * 1.2815516 * marginal).float(), atol=1e-4)


def test_low_rank_gaussian_rescales_exactly() -> None:
    objective = LowRankGaussianObjective(rank=2)
    parameters = _low_rank_parameters(1, 2, 3, 2)
    targets = torch.randn(1, 2, 3)
    location = torch.tensor([[[1.0, -2.0, 3.0]]])
    scale = torch.tensor([[[2.0, 0.5, 4.0]]])
    rescaled = objective.rescale(parameters, location, scale)
    # Change of variables: log p(s x + l) = log p(x) - sum log s.
    expected = objective.compute_log_likelihood(parameters, targets) - scale.log().sum()
    assert torch.allclose(objective.compute_log_likelihood(rescaled, targets * scale + location), expected, atol=1e-4)
    with pytest.raises(ValueError, match="rank"):
        _ = LowRankGaussianObjective(rank=0)


def test_sample_objective_statistics() -> None:
    objective = SampleObjective(num_samples=5, prediction_quantiles=(0.1, 0.5, 0.9))
    parameters = torch.tensor([[[[4.0, 1.0, 3.0, 2.0, 10.0]]]])
    assert objective.compute_point(parameters).item() == 3.0
    mean_objective = SampleObjective(num_samples=5, point_estimate=SamplePointEstimate.MEAN)
    assert mean_objective.compute_point(parameters).item() == 4.0
    expected = np.quantile(parameters.flatten().numpy(), [0.1, 0.5, 0.9])
    assert objective.compute_quantiles(parameters).flatten().numpy() == pytest.approx(expected)
    assert objective.level_parameters == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="at least 2"):
        _ = SampleObjective(num_samples=1)


def test_sample_objective_loss_is_the_fair_crps() -> None:
    objective = SampleObjective(num_samples=4)
    _ = torch.manual_seed(0)
    parameters = torch.randn(2, 3, 2, 4)
    targets = torch.randn(2, 3, 2)
    error = (parameters - targets.unsqueeze(-1)).abs().mean(-1)
    pairs = (parameters.unsqueeze(-1) - parameters.unsqueeze(-2)).abs().sum((-1, -2)) / (2 * 4 * 3)
    assert objective.compute_loss(parameters, targets).item() == pytest.approx(float((error - pairs).mean()), rel=1e-5)


def test_metrics_crps_and_coverage() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(3, 7, 2, 1))
    targets = rng.normal(size=(3, 2, 1))
    point = np.median(samples, axis=1)
    prediction = ForecastPrediction(
        point=point,
        quantile_levels=(0.25, 0.5, 0.75),
        quantiles=np.moveaxis(np.quantile(samples, [0.25, 0.5, 0.75], axis=1), 0, -1),
        samples=samples,
    )
    metrics = ForecastMetrics.compute(prediction, targets)
    values = np.moveaxis(samples, 1, -1)
    brute = np.abs(values - targets[..., None]).mean(-1) - 0.5 * np.abs(
        values[..., :, None] - values[..., None, :]
    ).mean((-1, -2))
    assert metrics.crps == pytest.approx(float(brute.mean()))
    assert metrics.quantile_loss is not None
    assert metrics.quantile_crps == pytest.approx(2.0 * metrics.quantile_loss)
    lower = prediction.quantiles[..., 0]
    upper = prediction.quantiles[..., 2]
    assert metrics.interval_coverage is not None and metrics.interval_width is not None
    assert list(metrics.interval_coverage) == [0.5]
    assert metrics.interval_coverage[0.5] == pytest.approx(float(np.mean((targets >= lower) & (targets <= upper))))
    assert metrics.interval_width[0.5] == pytest.approx(float(np.mean(upper - lower)))
    assert metrics.quantile_coverage is not None
    assert metrics.quantile_coverage[0.5] == pytest.approx(float(np.mean(targets <= point)))
    point_metrics = ForecastMetrics.compute(
        ForecastPrediction(point=point, quantile_levels=(), quantiles=point[..., :0], samples=point[:, None]), targets
    )
    assert point_metrics.crps == pytest.approx(point_metrics.mae)
    assert point_metrics.quantile_coverage is None and point_metrics.interval_coverage is None
