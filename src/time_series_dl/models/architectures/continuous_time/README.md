# continuous_time

## Overview

Models of irregularly sampled sequences, which read observation times, values and observed masks instead of regular
arrays: attention over learned time embeddings, and neural ordinary and controlled differential equations solved
with the fixed-step solvers of [differential_equations/](../../components/differential_equations/README.md). Their
networks implement the irregular interfaces, so they serve the tasks of [tasks/irregular/](../../../tasks/irregular/README.md)
rather than the tasks of regular arrays.

## Components

| Component | Parameters | Classification / regression | Interpolation / forecasting |
| --------- | ---------- | :-------------------------: | :-------------------------: |
| [mTAN (Shukla and Marlin, 2021)](./mtan/README.md) | `MTANParameters` | ✓ | ✓ |
| [Neural CDE (Kidger et al., 2020)](./neural_cde/README.md) | `NeuralCDEParameters` | ✓ | |
| [ODE-RNN (Rubanova et al., 2019)](./ode_rnn/README.md) | `ODERNNParameters` | ✓ | |
| [Latent ODE (Rubanova et al., 2019)](./latent_ode/README.md) | `LatentODEParameters` | ✓ | ✓ |

## Examples

```python
from time_series_dl.models.architectures.continuous_time import NeuralCDEParameters
from time_series_dl.models.components.differential_equations import CubicInterpolationKind
from time_series_dl.tasks.irregular.classification import IrregularClassifierSettings

classifier = IrregularClassifierSettings(
    model=NeuralCDEParameters(interpolation=CubicInterpolationKind.NATURAL), is_class_balanced=True
).build()
```
