# time_gan

## Overview

TimeGAN (Yoon, Jarrett and van der Schaar, "Time-series Generative Adversarial Networks", NeurIPS 2019): an
embedder and a recovery network form a recurrent autoencoder, a generator and a supervisor produce latent sequences
from noise, and a discriminator tells real latent sequences from generated ones. It is a sequence generation model
for the generation task.

Training runs three phases over the epochs of the shared trainer, set by `TimeGANPhaseSchedule`:

| Epochs | Phase | Trained networks and loss |
| ------ | ----- | ------------------------- |
| `[0, autoencoder_epochs)` | autoencoder | embedder and recovery: reconstruction |
| `[autoencoder_epochs, autoencoder_epochs + supervised_epochs)` | supervised | supervisor: next latent step of real sequences |
| from `autoencoder_epochs + supervised_epochs` up to `max_epochs` | joint | generator and supervisor: adversarial, supervised and moment losses; embedder and recovery: reconstruction and supervised loss; discriminator: classification loss while it exceeds a threshold |

Differences from the paper and the authors' TensorFlow code: one optimizer takes one step of every network per batch
(the authors alternate optimizers with two generator and embedder steps per discriminator step), with gradients
routed through frozen sub-networks so each network gets exactly its own phase gradient; phase lengths are counted in
epochs rather than iterations; the discriminator is unidirectional, as in the authors' code (the paper describes a
bidirectional one); the trainer monitors the moment loss in the joint phase, and every earlier epoch counts as an
improvement so early stopping and best-epoch selection act within the joint phase only; and all sequences have the
same length.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimeGANParameters`: cell, sizes, noise size, phase schedule, gamma and discriminator threshold. |
| [phase.py](./phase.py) | `TimeGANPhase`: autoencoder, supervised or joint training. |
| [phase_schedule.py](./phase_schedule.py) | `TimeGANPhaseSchedule`: the phase of each epoch. |
| [recurrent_block.py](./recurrent_block.py) | `TimeGANRecurrentBlock`: recurrent layers, a linear map of every step and an optional sigmoid. |
| [sequence_generation_network.py](./sequence_generation_network.py) | `TimeGANSequenceGenerationNetwork`: the five networks, the phase losses through one optimizer, and sampling. |
