# generation

## Overview

Synthetic time series generation: a generative model learns the distribution of fixed-length multivariate sequences
(given directly, or as windows of long series) and draws new ones in the original scale. Any
`SequenceGenerationModel` plugs in, such as TimeGAN or Diffusion-TS; each trains with its own objective through the
shared trainer. Synthetic sequences are evaluated against real ones with TimeGAN's discriminative and predictive
scores and two cheap distribution distances.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `GeneratorSettings`: model, training, scaling, windows of long series and evaluation. |
| [generator.py](./generator.py) | `SequenceGenerator`: `fit`, `fit_series`, `generate`, `evaluate`, `save` and `load`. |
| [generation_step.py](./generation_step.py) | `GenerationStep`: a generation network's loss of a batch, with the epoch. |
| [metrics.py](./metrics.py) | `GenerationMetrics`: discriminative and predictive scores, marginal and autocorrelation distances. |
| [evaluation/](./evaluation/README.md) | The individual metrics and their post-hoc networks. |

## Examples

```python
from time_series_dl.models.architectures.recurrent.time_gan import TimeGANParameters
from time_series_dl.tasks.generation import GeneratorSettings

generator = GeneratorSettings(model=TimeGANParameters(), window_length=24).build()
generator.fit_series(series)
samples = generator.generate(1000, random_seed=0)  # (1000, 24, channels)
metrics = generator.evaluate(real_windows)
```
