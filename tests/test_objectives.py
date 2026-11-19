import math

import pytest
import torch

from time_series_dl import HuberLoss, LogCoshLoss, MAELoss, MSELoss, PinballLoss, PointLoss
from time_series_dl.tasks.forecasting import (
    ForecastMetrics,
    ForecastPrediction,
    GaussianObjective,
    LaplaceObjective,
    PointObjective,
    QuantileObjective,
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
    import numpy as np

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
