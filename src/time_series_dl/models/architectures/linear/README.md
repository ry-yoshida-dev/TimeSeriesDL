# linear

## Overview

Models that map the window to the horizon with linear layers over the time axis or its spectrum.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation | Anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :--------: | :---------------: |
| [DLinear (Zeng et al., 2023)](./dlinear/README.md) | `DLinearParameters` | ✓ | ✓ | ✓ | ✓ |
| [NLinear (Zeng et al., 2023)](./nlinear/README.md) | `NLinearParameters` | ✓ | | | |
| [RLinear (Li et al., 2023)](./rlinear/README.md) | `RLinearParameters` | ✓ | | ✓ | ✓ |
| [FITS (Xu et al., 2024)](./fits/README.md) | `FITSParameters` | ✓ | | ✓ | ✓ |
| [SparseTSF (Lin et al., 2024)](./sparse_tsf/README.md) | `SparseTSFParameters` | ✓ | | | |
| [CycleNet (Lin et al., 2024)](./cycle_net/README.md) | `CycleNetParameters` | ✓ (with a phase covariate) | | | |
