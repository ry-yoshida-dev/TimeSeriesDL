import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from .epoch_scheduled_objective import EpochScheduledObjective
from .history import TrainingHistory
from .objective import TrainingObjective
from .settings import TrainingSettings


@dataclass(frozen=True)
class Trainer:
    """
    Mini-batch training loop shared by every task.

    Attributes
    ----------
    settings : TrainingSettings
        Optimizer, schedule, early stopping and device.
    """

    settings: TrainingSettings

    @property
    def device(self) -> torch.device:
        """
        Get the device that training runs on.

        Returns
        -------
        torch.device
            The resolved device.
        """
        return self.settings.device.resolve()

    def seed(self) -> None:
        """
        Seed PyTorch's global generator, before networks are built, when a seed is set.
        """
        if self.settings.random_seed is not None:
            _ = torch.manual_seed(self.settings.random_seed)

    def fit(
        self,
        network: nn.Module,
        objective: TrainingObjective,
        training_data: TensorDataset,
        validation_data: TensorDataset | None = None,
    ) -> TrainingHistory:
        """
        Train a network in place, keeping the weights of the best epoch.

        Parameters
        ----------
        network : nn.Module
            The network whose parameters the objective's loss depends on; moved to the training device.
        objective : TrainingObjective
            The loss of one batch; an ``EpochScheduledObjective`` is told each epoch before it starts.
        training_data : TensorDataset
            The training samples.
        validation_data : TensorDataset | None
            Samples monitored for early stopping and best-epoch selection; the training loss is monitored when None.

        Returns
        -------
        TrainingHistory
            The losses of every epoch.

        Raises
        ------
        ValueError
            If the training or validation data is empty, or the network has no trainable parameters.
        FloatingPointError
            If the training loss becomes NaN or infinite.
        """
        if len(training_data) == 0:
            raise ValueError("training_data must not be empty.")
        if validation_data is not None and len(validation_data) == 0:
            raise ValueError("validation_data must not be empty when given.")
        trainable: list[nn.Parameter] = [parameter for parameter in network.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("network has no trainable parameters.")
        settings: TrainingSettings = self.settings
        device: torch.device = self.device
        _ = network.to(device)
        generator: torch.Generator = torch.Generator()
        if settings.random_seed is not None:
            _ = generator.manual_seed(settings.random_seed)
        loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            training_data, batch_size=settings.batch_size, shuffle=True, generator=generator
        )
        optimizer: torch.optim.Optimizer = settings.optimizer.build(
            trainable, settings.learning_rate, settings.weight_decay
        )
        scheduler: LRScheduler = settings.schedule.build(
            optimizer, settings.learning_rate, settings.max_epochs, len(loader)
        )
        training_losses: list[float] = []
        validation_losses: list[float] = []
        best_loss: float = math.inf
        best_epoch: int = 0
        best_state: dict[str, Tensor] = copy.deepcopy(network.state_dict())
        epochs_without_improvement: int = 0
        is_stopped_early: bool = False
        for epoch in range(settings.max_epochs):
            if isinstance(objective, EpochScheduledObjective):
                objective.start_epoch(epoch)
            training_loss: float = self._train_epoch(network, objective, loader, optimizer, scheduler)
            training_losses.append(training_loss)
            if not settings.schedule.is_stepped_per_batch:
                scheduler.step()
            monitored_loss: float = training_loss
            if validation_data is not None:
                monitored_loss = self.compute_loss(network, objective, validation_data)
                validation_losses.append(monitored_loss)
            if settings.is_verbose:
                progress: str = f"epoch {epoch + 1}/{settings.max_epochs}"
                print(f"{progress}: training {training_loss:.6g}, monitored {monitored_loss:.6g}")
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_epoch = epoch
                best_state = copy.deepcopy(network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            patience: int | None = settings.early_stopping_patience
            if patience is not None and epochs_without_improvement >= patience:
                is_stopped_early = epoch + 1 < settings.max_epochs
                break
        _ = network.load_state_dict(best_state)
        _ = network.eval()
        return TrainingHistory(
            training_losses=tuple(training_losses),
            validation_losses=tuple(validation_losses),
            best_epoch=best_epoch,
            is_stopped_early=is_stopped_early,
        )

    def compute_loss(self, network: nn.Module, objective: TrainingObjective, data: TensorDataset) -> float:
        """
        Compute the sample-weighted mean evaluation loss of a dataset.

        Parameters
        ----------
        network : nn.Module
            The network, already on the training device.
        objective : TrainingObjective
            The loss of one batch.
        data : TensorDataset
            The samples.

        Returns
        -------
        float
            The mean loss.
        """
        _ = network.eval()
        loader: DataLoader[tuple[Tensor, ...]] = DataLoader(data, batch_size=self.settings.batch_size)
        total: float = 0.0
        count: int = 0
        with torch.no_grad():
            for batch in loader:
                moved: tuple[Tensor, ...] = self._move(batch)
                loss: Tensor = objective.compute_loss(moved, is_training=False)
                size: int = moved[0].shape[0]
                total += float(loss.item()) * size
                count += size
        return total / count

    def _train_epoch(
        self,
        network: nn.Module,
        objective: TrainingObjective,
        loader: DataLoader[tuple[Tensor, ...]],
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
    ) -> float:
        _ = network.train()
        total: float = 0.0
        count: int = 0
        for batch in loader:
            moved: tuple[Tensor, ...] = self._move(batch)
            optimizer.zero_grad(set_to_none=True)
            loss: Tensor = objective.compute_loss(moved, is_training=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"training loss is not finite: {loss.item()}")
            loss.backward()
            if self.settings.gradient_clip_norm is not None:
                _ = nn.utils.clip_grad_norm_(network.parameters(), self.settings.gradient_clip_norm)
            optimizer.step()
            if self.settings.schedule.is_stepped_per_batch:
                scheduler.step()
            size: int = moved[0].shape[0]
            total += float(loss.item()) * size
            count += size
        return total / count

    def _move(self, batch: tuple[Tensor, ...] | list[Tensor]) -> tuple[Tensor, ...]:
        device: torch.device = self.device
        return tuple(tensor.to(device) for tensor in batch)
