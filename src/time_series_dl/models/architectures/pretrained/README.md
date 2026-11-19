# pretrained

## Overview

Models around the encoder of another model: self-supervised pretraining methods that wrap any encoder, and the
transfer of a pretrained encoder to the networks of every task. The pretraining methods are trained by
`RepresentationLearner`; TS2Vec and TS-TCC, which come with their own encoders, are convolutional models.

## Components

| Component | Parameters | Pretraining | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :---------: | :-------------------------: | :----------------------------: |
| [SimMTM (Dong et al., 2023)](./sim_mtm/README.md) | `SimMTMParameters` | ✓ | | | |
| [Masked reconstruction (Zerveas et al., 2021; Nie et al., 2023)](./masked_reconstruction/README.md) | `MaskedReconstructionParameters` | ✓ | | | |
| [Pretrained encoder](./encoder/README.md) | `PretrainedEncoderParameters` | | ✓ | ✓ | ✓ |
