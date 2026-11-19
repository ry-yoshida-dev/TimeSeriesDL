"""Shape, loss, gradient and sampling checks of the sequence generation networks of every generation model."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from time_series_dl.models import SequenceShape
from time_series_dl.models.interfaces.protocols.sequence_generation import SequenceGenerationModel
from time_series_dl.models.interfaces.sequence_generation_network import SequenceGenerationNetwork

BATCH_SIZE: int = 3


def check_sequence_generation_network(
    model: SequenceGenerationModel, shape: SequenceShape, epochs: Sequence[int] = (0,)
) -> SequenceGenerationNetwork:
    """
    Check the loss of a sequence generation network in the given epochs, that every parameter receives a gradient
    in the last of them, and that seeded samples have the right shape and are reproducible.

    Parameters
    ----------
    model : SequenceGenerationModel
        The model under test.
    shape : SequenceShape
        The dimensions to build the network with.
    epochs : Sequence[int]
        Epochs whose loss is checked; every parameter must receive a gradient in the last one.

    Returns
    -------
    SequenceGenerationNetwork
        The network, in evaluation mode.
    """
    _ = torch.manual_seed(0)
    network: SequenceGenerationNetwork = model.build_sequence_generation_network(shape)
    assert network.minimum_epochs >= 1
    values: Tensor = torch.rand(BATCH_SIZE, shape.length, shape.channels)
    _ = network.train()
    for epoch in epochs:
        network.zero_grad()
        loss: Tensor = network.compute_loss(values, epoch)
        assert loss.shape == ()
        assert bool(torch.isfinite(loss))
        loss.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient in epoch {epochs[-1]}: {missing}"
    _ = network.eval()
    with torch.no_grad():
        first: Tensor = network.sample(BATCH_SIZE + 1, torch.Generator().manual_seed(3))
        second: Tensor = network.sample(BATCH_SIZE + 1, torch.Generator().manual_seed(3))
        other: Tensor = network.sample(BATCH_SIZE + 1, torch.Generator().manual_seed(4))
    assert first.shape == (BATCH_SIZE + 1, shape.length, shape.channels)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    return network


def collect_gradients(loss: Tensor, network: nn.Module, prefix: str) -> list[Tensor]:
    """
    Compute the gradient of a loss with respect to the parameters whose names start with a prefix.

    Parameters
    ----------
    loss : Tensor
        The scalar loss.
    network : nn.Module
        The network.
    prefix : str
        The prefix of the parameter names, such as the name of a sub-network.

    Returns
    -------
    list[Tensor]
        The gradients in parameter order; zeros for parameters the loss does not depend on.
    """
    parameters: list[nn.Parameter] = [
        parameter for name, parameter in network.named_parameters() if name.startswith(prefix)
    ]
    assert parameters, f"no parameters start with {prefix}"
    gradients: tuple[Tensor, ...] = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True, materialize_grads=True
    )
    return list(gradients)


def assert_gradients_close(actual: list[Tensor], expected: list[Tensor]) -> None:
    """
    Assert that two lists of gradients match.

    Parameters
    ----------
    actual : list[Tensor]
        The gradients of the combined loss.
    expected : list[Tensor]
        The gradients of the paper's losses.
    """
    assert len(actual) == len(expected)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-4)
