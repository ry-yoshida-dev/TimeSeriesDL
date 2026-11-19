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
| [adversarial/](./adversarial/README.md) | Gradient routing for adversarial training through one optimizer. |
| [differential_equations/](./differential_equations/README.md) | Fixed-step ODE solvers and cubic control paths of continuous-time models. |
| [irregular_heads/](./irregular_heads/README.md) | Output layers that turn an irregular sequence encoder into a summary network. |
| [diffusion/](./diffusion/README.md) | Noise schedules, forward process, ancestral sampler and step embedding of denoising diffusion models. |
| [pretrained/](./pretrained/README.md) | Download and strict loading of published weights. |
| [t5/](./t5/README.md) | T5 encoder and decoder stacks of Chronos, Chronos-Bolt and MOMENT. |
| [covariates/](./covariates/README.md) | Static covariate embedding and the generic covariate adapter of forecast networks. |
