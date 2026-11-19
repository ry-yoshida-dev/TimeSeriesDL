"""Shape and gradient checks shared by the tests of every model."""

import torch
from torch import Tensor

from time_series_dl.models import (
    ForecastingModel,
    ForecastShape,
    ReconstructionModel,
    SequenceShape,
    SummaryModel,
    SummaryShape,
)

BATCH_SIZE: int = 3


def check_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    """
    Check the output shape of a forecast network and that every parameter receives a gradient.

    Parameters
    ----------
    model : ForecastingModel
        The model under test.
    shape : ForecastShape
        The dimensions to build the network with.
    """
    _ = torch.manual_seed(0)
    network = model.build_forecast_network(shape)
    past: Tensor = torch.randn(BATCH_SIZE, shape.input_length, shape.input_channels)
    future: Tensor = torch.randn(BATCH_SIZE, shape.horizon, shape.output_channels)
    _ = network.train()
    outputs: Tensor = network(past, future)
    assert outputs.shape == (BATCH_SIZE, shape.horizon, shape.output_channels, shape.num_parameters)
    outputs.square().mean().backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    with torch.no_grad():
        inferred: Tensor = network(past)
    assert inferred.shape == outputs.shape
    assert bool(torch.isfinite(inferred).all())


def check_reconstruction_network(model: ReconstructionModel, shape: SequenceShape) -> None:
    """
    Check the output shape of a reconstruction network and that every parameter receives a gradient.

    Parameters
    ----------
    model : ReconstructionModel
        The model under test.
    shape : SequenceShape
        The dimensions to build the network with.
    """
    _ = torch.manual_seed(0)
    network = model.build_reconstruction_network(shape)
    values: Tensor = torch.randn(BATCH_SIZE, shape.length, shape.channels)
    _ = network.train()
    outputs: Tensor = network(values)
    assert outputs.shape == values.shape
    outputs.square().mean().backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def check_summary_network(model: SummaryModel, shape: SummaryShape) -> None:
    """
    Check the output shape of a summary network and that every parameter receives a gradient.

    Parameters
    ----------
    model : SummaryModel
        The model under test.
    shape : SummaryShape
        The dimensions to build the network with.
    """
    _ = torch.manual_seed(0)
    network = model.build_summary_network(shape)
    values: Tensor = torch.randn(BATCH_SIZE, shape.input_length, shape.input_channels)
    _ = network.train()
    outputs: Tensor = network(values)
    assert outputs.shape == (BATCH_SIZE, shape.output_size)
    outputs.square().mean().backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
