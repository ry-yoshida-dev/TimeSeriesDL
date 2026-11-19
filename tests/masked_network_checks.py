"""Shape, gradient and mask checks shared by the tests of masked reconstruction networks."""

import torch
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.interfaces.masked_reconstruction import MaskedReconstruction
from time_series_dl.models.interfaces.protocols.masked_reconstruction import MaskedReconstructionModel

BATCH_SIZE: int = 3


def check_masked_reconstruction_network(model: MaskedReconstructionModel, shape: SequenceShape) -> None:
    """
    Check the output shape of a masked reconstruction network and that every parameter receives a gradient.

    Parameters
    ----------
    model : MaskedReconstructionModel
        The model under test.
    shape : SequenceShape
        The dimensions to build the network with.
    """
    _ = torch.manual_seed(0)
    network = model.build_masked_reconstruction_network(shape)
    is_observed: Tensor = torch.rand(BATCH_SIZE, shape.length, shape.channels) < 0.7
    values: Tensor = torch.randn(BATCH_SIZE, shape.length, shape.channels) * is_observed
    _ = network.train()
    output: MaskedReconstruction = network(values, is_observed)
    assert output.values.shape == values.shape
    assert network.hidden_value_loss_weight >= 0.0
    objective: Tensor = output.values.square().mean()
    if output.loss is not None:
        assert bool(torch.isfinite(output.loss))
        objective = objective + output.loss
    objective.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    with torch.no_grad():
        inferred: Tensor = network(values, is_observed).values
    assert bool(torch.isfinite(inferred).all())


def check_reads_observed_mask(model: MaskedReconstructionModel, shape: SequenceShape) -> None:
    """
    Check that a zero marked missing is reconstructed differently from the same zero marked observed.

    Parameters
    ----------
    model : MaskedReconstructionModel
        The model under test.
    shape : SequenceShape
        The dimensions to build the network with; it needs at least two time steps and two channels.
    """
    _ = torch.manual_seed(0)
    network = model.build_masked_reconstruction_network(shape).eval()
    values: Tensor = torch.randn(BATCH_SIZE, shape.length, shape.channels)
    values[:, 1, 1] = 0.0
    is_observed: Tensor = torch.ones(values.shape, dtype=torch.bool)
    is_missing_zero: Tensor = is_observed.clone()
    is_missing_zero[:, 1, 1] = False
    with torch.no_grad():
        observed_zero: Tensor = network(values, is_observed).values
        missing_zero: Tensor = network(values, is_missing_zero).values
    assert not torch.allclose(observed_zero, missing_zero, atol=1e-6)
