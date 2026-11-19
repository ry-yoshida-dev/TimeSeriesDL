# mlp

## Overview

Models built from multilayer perceptrons, including basis-expansion and mixing architectures.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: |
| [MLP](./vanilla/README.md) | `MLPParameters` | ✓ | ✓ | ✓ |
| [N-BEATS (Oreshkin et al., 2020)](./n_beats/README.md) | `NBEATSParameters` | ✓ | | |
| [N-HiTS (Challu et al., 2023)](./n_hits/README.md) | `NHITSParameters` | ✓ | | |
| [TSMixer (Chen et al., 2023)](./ts_mixer/README.md) | `TSMixerParameters` | ✓ | ✓ | ✓ |
| [TiDE (Das et al., 2023)](./tide/README.md) | `TiDEParameters` | ✓ | | |
