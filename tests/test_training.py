import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from time_series_dl import Device, LearningRateSchedule, OptimizerKind, TrainingSettings
from time_series_dl.training import Trainer


class LinearObjective:
    def __init__(self, network: nn.Linear) -> None:
        self.network = network

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        prediction: Tensor = self.network(batch[0])
        return (prediction - batch[1]).square().mean()


def make_data(size: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(size, 2, generator=generator)
    return TensorDataset(inputs, inputs @ torch.tensor([[2.0], [-1.0]]))


@pytest.mark.parametrize("optimizer", list(OptimizerKind))
@pytest.mark.parametrize("schedule", list(LearningRateSchedule))
def test_trainer_reduces_loss(optimizer: OptimizerKind, schedule: LearningRateSchedule) -> None:
    settings = TrainingSettings(
        max_epochs=30,
        batch_size=8,
        learning_rate=0.05,
        optimizer=optimizer,
        schedule=schedule,
        device=Device.CPU,
        early_stopping_patience=None,
        gradient_clip_norm=10.0,
    )
    network = nn.Linear(2, 1)
    history = Trainer(settings).fit(network, LinearObjective(network), make_data(64))
    assert history.training_losses[-1] < history.training_losses[0]
    assert history.best_loss == min(history.training_losses)
    assert not history.is_stopped_early


def test_trainer_stops_early_and_restores_best_weights() -> None:
    settings = TrainingSettings(
        max_epochs=200, learning_rate=1.0, optimizer=OptimizerKind.SGD, early_stopping_patience=2, device=Device.CPU
    )
    network = nn.Linear(2, 1)
    trainer = Trainer(settings)
    validation = make_data(16)
    history = trainer.fit(network, LinearObjective(network), make_data(64), validation)
    assert history.is_stopped_early
    assert history.num_epochs == history.best_epoch + 3
    assert trainer.compute_loss(network, LinearObjective(network), validation) == pytest.approx(history.best_loss)


def test_trainer_rejects_diverging_loss() -> None:
    network = nn.Linear(2, 1)
    _ = network.weight.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        _ = Trainer(TrainingSettings(device=Device.CPU)).fit(network, LinearObjective(network), make_data(8))


def test_training_settings_validate() -> None:
    with pytest.raises(ValueError, match="max_epochs"):
        TrainingSettings(max_epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingSettings(learning_rate=0.0)
    with pytest.raises(ValueError, match="early_stopping_patience"):
        TrainingSettings(early_stopping_patience=0)
