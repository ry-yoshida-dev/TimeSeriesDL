# components

## Overview

Building blocks shared by several model families. Encoder-based models define a `SequenceEncoder` and get every task
from the generic heads.

## Components

| Component | Description |
| --------- | ----------- |
| [layers/](./layers/README.md) | Shared layers and options. |
| [attention/](./attention/README.md) | Multi-head attention with full, ProbSparse, auto-correlation and Fourier mechanisms. |
| [state_space/](./state_space/README.md) | HiPPO memories, kernels, scans and layers of the state space models. |
| [heads/](./heads/README.md) | Generic heads that turn an encoder into a network of every task. |
| [covariates/](./covariates/README.md) | Static covariate embedding and the generic covariate adapter of forecast networks. |
