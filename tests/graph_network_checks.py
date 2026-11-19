"""Checks shared by the tests of the graph neural networks, whose nodes are the channels of a series."""

import torch
from torch import Tensor

from time_series_dl.models import ForecastingModel, ForecastShape


def check_target_selection(model: ForecastingModel) -> None:
    """
    Check that forecasting some targets gives the forecasts of those channels when every channel is forecast.

    Parameters
    ----------
    model : ForecastingModel
        The model under test; its output layer must not depend on the number of targets.
    """
    every_target: ForecastShape = ForecastShape(
        input_length=14, input_channels=4, horizon=3, target_channels=(0, 1, 2, 3)
    )
    some_targets: ForecastShape = ForecastShape(input_length=14, input_channels=4, horizon=3, target_channels=(3, 1))
    _ = torch.manual_seed(0)
    every_network = model.build_forecast_network(every_target).eval()
    _ = torch.manual_seed(0)
    some_network = model.build_forecast_network(some_targets).eval()
    past: Tensor = torch.randn(2, 14, 4)
    with torch.no_grad():
        every_output: Tensor = every_network(past)
        some_output: Tensor = some_network(past)
    assert torch.allclose(every_output[:, :, [3, 1]], some_output, atol=1e-5)


def check_channel_mixing(model: ForecastingModel) -> None:
    """
    Check that a channel that is not forecast still changes the forecast of the targets through the graph.

    Parameters
    ----------
    model : ForecastingModel
        The model under test.
    """
    shape: ForecastShape = ForecastShape(input_length=14, input_channels=3, horizon=2, target_channels=(0,))
    _ = torch.manual_seed(0)
    network = model.build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 14, 3)
    changed: Tensor = past.clone()
    changed[:, :, 2] += torch.randn(2, 14)
    with torch.no_grad():
        original: Tensor = network(past)
        perturbed: Tensor = network(changed)
    assert not torch.allclose(original, perturbed, atol=1e-6)
