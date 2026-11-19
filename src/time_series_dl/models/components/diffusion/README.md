# diffusion

## Overview

Building blocks of denoising diffusion models shared by the generative forecasters (TimeGrad, CSDI): noise
schedules, the forward noising process with its ancestral sampler, and the embedding of the diffusion step.

## Components

| Component | Description |
| --------- | ----------- |
| [noise_schedule.py](./noise_schedule.py) | `NoiseSchedule`: kind, number of steps and first and last noise variances; builds the process. |
| [noise_schedule_kind.py](./noise_schedule_kind.py) | `NoiseScheduleKind`: linear or quadratic growth of the variances. |
| [diffusion_process.py](./diffusion_process.py) | `DiffusionProcess`: diffusion of clean values to a step, and sampling by the reverse process from any noise predictor. |
| [noised_sample.py](./noised_sample.py) | `NoisedSample`: noisy values, their noise and their steps, for the denoising loss. |
| [step_embedding.py](./step_embedding.py) | `DiffusionStepEmbedding`: sinusoidal features of the step and two SiLU layers. |

## Examples

```python
import torch

from time_series_dl.models.components.diffusion import NoiseSchedule, NoiseScheduleKind

process = NoiseSchedule(kind=NoiseScheduleKind.QUADRATIC, num_steps=50, last_variance=0.5).build_process()
noised = process.corrupt(clean)  # clean: (batch, ...)
loss = (denoiser(noised.noisy, noised.steps) - noised.noise).square().mean()
samples = process.sample(denoiser, tuple(clean.shape), torch.Generator().manual_seed(0))
```
