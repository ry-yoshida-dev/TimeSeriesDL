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
| [normalization/](./normalization/README.md) | Reversible instance normalization that is part of a model's architecture. |
| [random_convolution/](./random_convolution/README.md) | Fixed-kernel transforms, PPV/max/MPV/MIPV/LSPV pooling and the linear head of the ROCKET family. |
| [graph/](./graph/README.md) | Graph learning layers and graph convolutions of the graph neural networks. |
| [missing_values/](./missing_values/README.md) | Time gaps, temporal decay and last observation carried forward of the recurrent imputation models. |
| [masked_modeling/](./masked_modeling/README.md) | Random masks and the decoder of masked-modeling pretraining. |
| [covariates/](./covariates/README.md) | Static covariate embedding and the generic covariate adapter of forecast networks. |
