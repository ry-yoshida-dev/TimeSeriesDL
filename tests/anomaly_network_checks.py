"""Shape, loss and gradient checks of the anomaly scoring networks of every anomaly detection model."""

import torch
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from time_series_dl.models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel

BATCH_SIZE: int = 3


def check_anomaly_scoring_network(model: AnomalyScoringModel, shape: SequenceShape) -> AnomalyScoringNetwork:
    """
    Check the scores and the loss of an anomaly scoring network, and that every parameter receives a gradient in the
    first and a later epoch.

    Parameters
    ----------
    model : AnomalyScoringModel
        The model under test.
    shape : SequenceShape
        The dimensions to build the network with.

    Returns
    -------
    AnomalyScoringNetwork
        The network, in evaluation mode.
    """
    _ = torch.manual_seed(0)
    network: AnomalyScoringNetwork = model.build_anomaly_scoring_network(shape)
    values: Tensor = torch.rand(BATCH_SIZE, shape.length, shape.channels)
    _ = network.train()
    for epoch in (0, 4):
        network.zero_grad()
        loss: Tensor = network.compute_loss(values, epoch)
        assert loss.shape == ()
        assert bool(torch.isfinite(loss))
        loss.backward()
        missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
        assert not missing, f"parameters without gradient in epoch {epoch}: {missing}"
    _ = network.eval()
    with torch.no_grad():
        scores: Tensor = network(values)
    assert scores.shape == (BATCH_SIZE, shape.length)
    assert bool(torch.isfinite(scores).all())
    return network


def collect_gradients(loss: Tensor, network: AnomalyScoringNetwork, prefix: str) -> list[Tensor]:
    """
    Compute the gradient of a loss with respect to the parameters whose names start with a prefix.

    Parameters
    ----------
    loss : Tensor
        The scalar loss.
    network : AnomalyScoringNetwork
        The network.
    prefix : str
        The prefix of the parameter names, such as the name of a sub-network.

    Returns
    -------
    list[Tensor]
        The gradients in parameter order; zeros for parameters the loss does not depend on.
    """
    parameters: list[torch.nn.Parameter] = [
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
        The gradients of the paper's phase losses.
    """
    assert len(actual) == len(expected)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-4)
