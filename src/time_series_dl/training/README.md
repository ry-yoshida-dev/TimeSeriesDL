# training

## Overview

Mini-batch training shared by every task: the settings, the optimizer and learning-rate schedule choices, the
loop with early stopping and best-epoch restoration, and the history it returns. Each task supplies the loss of a
batch as a `TrainingObjective`.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `TrainingSettings`: epochs, batch size, learning rate, optimizer, schedule, early stopping, device and seed. |
| [trainer.py](./trainer.py) | `Trainer`: trains a network on a `TensorDataset` and keeps the weights of the best epoch. |
| [history.py](./history.py) | `TrainingHistory`: training and validation losses of every epoch. |
| [objective.py](./objective.py) | `TrainingObjective`: structural type of the per-batch loss of a task. |
| [epoch_scheduled_objective.py](./epoch_scheduled_objective.py) | `EpochScheduledObjective`: a `TrainingObjective` told the index of each epoch before it starts. |
| [optimizer_kind.py](./optimizer_kind.py) | `OptimizerKind`: Adam, AdamW or SGD. |
| [learning_rate_schedule.py](./learning_rate_schedule.py) | `LearningRateSchedule`: constant, cosine or one-cycle. |
