# convolutional

## Overview

Models built from convolutions over the time axis, including the ROCKET family, whose fixed kernels feed a trained
linear layer.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation | Anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :--------: | :---------------: |
| [FCN (Wang et al., 2017)](./fcn/README.md) | `FCNParameters` | ✓ | ✓ | ✓ | ✓ |
| [ResNet (Wang et al., 2017)](./resnet/README.md) | `ResNetParameters` | ✓ | ✓ | ✓ | ✓ |
| [InceptionTime (Ismail Fawaz et al., 2020)](./inception_time/README.md) | `InceptionTimeParameters` | ✓ | ✓ | ✓ | ✓ |
| [TCN (Bai et al., 2018)](./tcn/README.md) | `TCNParameters` | ✓ | ✓ | ✓ | ✓ |
| [TimesNet (Wu et al., 2023)](./times_net/README.md) | `TimesNetParameters` | ✓ | ✓ | ✓ | ✓ |
| [MICN (Wang et al., 2023)](./micn/README.md) | `MICNParameters` | ✓ | ✓ | ✓ | ✓ |
| [ModernTCN (Luo and Wang, 2024)](./modern_tcn/README.md) | `ModernTCNParameters` | ✓ | ✓ | ✓ | ✓ |
| [LITE (Ismail-Fawaz et al., 2023)](./lite/README.md) | `LITEParameters` | ✓ | ✓ | ✓ | ✓ |
| [ROCKET (Dempster et al., 2020)](./rocket/README.md) | `RocketParameters` | | ✓ | | |
| [MiniRocket (Dempster et al., 2021)](./mini_rocket/README.md) | `MiniRocketParameters` | | ✓ | | |
| [MultiRocket (Tan et al., 2022)](./multi_rocket/README.md) | `MultiRocketParameters` | | ✓ | | |
