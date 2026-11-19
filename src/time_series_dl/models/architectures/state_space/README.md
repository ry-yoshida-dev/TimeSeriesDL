# state_space

## Overview

Models built from linear state space models: recurrent cells with a HiPPO memory, deep state space models that stack
linear time-invariant or selective layers, and forecasters built on Legendre memories or Mamba.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: |
| [HiPPO-RNN (Gu et al., 2020)](./hippo_rnn/README.md) | `HiPPORNNParameters` | ✓ | ✓ | ✓ |
| [LMU (Voelker et al., 2019)](./lmu/README.md) | `LMUParameters` | ✓ | ✓ | ✓ |
| [S4 (Gu et al., 2022)](./s4/README.md) | `S4Parameters` | ✓ | ✓ | ✓ |
| [S4D (Gu et al., 2022)](./s4d/README.md) | `S4DParameters` | ✓ | ✓ | ✓ |
| [S5 (Smith et al., 2023)](./s5/README.md) | `S5Parameters` | ✓ | ✓ | ✓ |
| [LRU (Orvieto et al., 2023)](./lru/README.md) | `LRUParameters` | ✓ | ✓ | ✓ |
| [Mamba (Gu and Dao, 2024)](./mamba/README.md) | `MambaParameters` | ✓ | ✓ | ✓ |
| [Mamba-2 (Dao and Gu, 2024)](./mamba2/README.md) | `Mamba2Parameters` | ✓ | ✓ | ✓ |
| [S-Mamba (Wang et al., 2025)](./s_mamba/README.md) | `SMambaParameters` | ✓ | ✓ | ✓ |
| [FiLM (Zhou et al., 2022)](./film/README.md) | `FiLMParameters` | ✓ | | |
| [Deep state space encoder](./deep_state_space/README.md) | Built by the S4, S4D, S5, LRU, Mamba and Mamba-2 parameters | ✓ | ✓ | ✓ |

## Examples

```python
from time_series_dl.models import HiPPOMeasure, HiPPORNNParameters, S4DParameters
from time_series_dl.tasks.classification import ClassifierSettings

classifier = ClassifierSettings(model=S4DParameters(is_bidirectional=True)).build()
memory_rnn = HiPPORNNParameters(measure=HiPPOMeasure.LEGT, window_ratio=0.5)
```
