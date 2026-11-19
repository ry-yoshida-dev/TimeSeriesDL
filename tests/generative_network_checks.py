"""Shape, gradient, determinism and training checks shared by the tests of every generative forecast model."""

import torch
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.interfaces.generative_forecast_network import GenerativeForecastNetwork
from time_series_dl.models.interfaces.protocols.generative_forecasting import GenerativeForecastingModel

BATCH_SIZE: int = 3
NUM_SAMPLES: int = 4


def check_generative_forecast_network(model: GenerativeForecastingModel, shape: ForecastShape) -> None:
    """
    Check the loss, the gradients, the sample shapes and the reproducibility of a generative forecast network.

    Parameters
    ----------
    model : GenerativeForecastingModel
        The model under test.
    shape : ForecastShape
        The dimensions to build the network with; ``num_parameters`` is the number of samples of ``forward``.
    """
    _ = torch.manual_seed(0)
    network: GenerativeForecastNetwork = model.build_generative_forecast_network(shape)
    assert isinstance(model.build_forecast_network(shape), GenerativeForecastNetwork)
    past: Tensor = torch.randn(BATCH_SIZE, shape.input_length, shape.input_channels)
    future: Tensor = torch.randn(BATCH_SIZE, shape.horizon, shape.output_channels)
    _ = network.train()
    loss: Tensor = network.compute_loss(past, future)
    assert loss.dim() == 0
    assert bool(torch.isfinite(loss))
    loss.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    with torch.no_grad():
        samples: Tensor = network.sample(past, NUM_SAMPLES, torch.Generator().manual_seed(1))
        repeated: Tensor = network.sample(past, NUM_SAMPLES, torch.Generator().manual_seed(1))
        other: Tensor = network.sample(past, NUM_SAMPLES, torch.Generator().manual_seed(2))
        outputs: Tensor = network(past)
    assert samples.shape == (BATCH_SIZE, NUM_SAMPLES, shape.horizon, shape.output_channels)
    assert bool(torch.isfinite(samples).all())
    assert torch.equal(samples, repeated)
    assert not torch.equal(samples, other)
    assert outputs.shape == (BATCH_SIZE, shape.horizon, shape.output_channels, shape.num_parameters)


def check_loss_decreases(model: GenerativeForecastingModel, num_steps: int = 80) -> None:
    """
    Check that the training loss of a generative forecast network decreases on windows of a noisy sine.

    Parameters
    ----------
    model : GenerativeForecastingModel
        The model under test.
    num_steps : int
        Number of optimizer steps.
    """
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=16, input_channels=2, horizon=4, target_channels=(0, 1))
    network: GenerativeForecastNetwork = model.build_generative_forecast_network(shape)
    starts: Tensor = torch.randint(0, 200, (32, 1))
    steps: Tensor = (starts + torch.arange(20)).float()
    windows: Tensor = torch.stack([torch.sin(steps / 4.0), torch.cos(steps / 6.0)], dim=-1)
    windows = windows + 0.05 * torch.randn(windows.shape)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    losses: list[float] = []
    _ = network.train()
    for _step in range(num_steps):
        optimizer.zero_grad()
        loss: Tensor = network.compute_loss(windows[:, :16], windows[:, 16:])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    window: int = num_steps // 4
    first: float = sum(losses[:window]) / window
    last: float = sum(losses[-window:]) / window
    assert last < 0.8 * first, f"loss did not decrease: {first} -> {last}"
