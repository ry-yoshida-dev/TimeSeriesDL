# mlp

## Overview

Models built from multilayer perceptrons, including basis-expansion, mixing, frequency-domain and Koopman architectures.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation | Anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :--------: | :---------------: |
| [MLP](./vanilla/README.md) | `MLPParameters` | ✓ | ✓ | ✓ | ✓ |
| [N-BEATS (Oreshkin et al., 2020)](./n_beats/README.md) | `NBEATSParameters` | ✓ | | | |
| [N-HiTS (Challu et al., 2023)](./n_hits/README.md) | `NHITSParameters` | ✓ | | | |
| [TSMixer (Chen et al., 2023)](./ts_mixer/README.md) | `TSMixerParameters` | ✓ | ✓ | ✓ | ✓ |
| [TiDE (Das et al., 2023)](./tide/README.md) | `TiDEParameters` | ✓ | | | |
| [TimeMixer (Wang et al., 2024)](./time_mixer/README.md) | `TimeMixerParameters` | ✓ | ✓ | ✓ | ✓ |
| [TimeMixer++ (Wang et al., 2025)](./time_mixer_plus_plus/README.md) | `TimeMixerPlusPlusParameters` | ✓ | ✓ | ✓ | ✓ |
| [FreTS (Yi et al., 2023)](./frets/README.md) | `FreTSParameters` | ✓ | | | |
| [Koopa (Liu et al., 2023)](./koopa/README.md) | `KoopaParameters` | ✓ | | | |
| [USAD (Audibert et al., 2020)](./usad/README.md) | `USADParameters` | | | | ✓ |
