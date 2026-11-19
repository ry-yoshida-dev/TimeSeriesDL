# architectures

## Overview

Every model, grouped by the family of its main building block. Each model directory holds its parameters class and
the networks it builds.

## Components

| Component | Description |
| --------- | ----------- |
| [linear/](./linear/README.md) | Linear models over the time axis or its spectrum. |
| [mlp/](./mlp/README.md) | Multilayer perceptron, MLP-mixing, frequency-domain and Koopman models. |
| [recurrent/](./recurrent/README.md) | Recurrent neural networks. |
| [convolutional/](./convolutional/README.md) | Convolutional networks. |
| [transformer/](./transformer/README.md) | Attention-based models. |
| [state_space/](./state_space/README.md) | State space models, from HiPPO memories to Mamba. |
| [graph/](./graph/README.md) | Graph neural networks over the channels as nodes, with learned graphs. |
| [pretrained/](./pretrained/README.md) | Self-supervised pretraining around any encoder, and transfer of pretrained encoders. |
| [continuous_time/](./continuous_time/README.md) | Models of irregularly sampled sequences: multi-time attention and neural ODEs and CDEs. |
