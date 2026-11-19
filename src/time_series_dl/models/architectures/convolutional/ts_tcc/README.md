# ts_tcc

## Overview

TS-TCC (Eldele et al., "Time-Series Representation Learning via Temporal and Contextual Contrasting", IJCAI 2021):
a three-block convolutional encoder is pretrained on a weak (scaling) and a strong (segment permutation and
jitter) augmentation of each sequence. Temporal contrasting summarizes the past features of one view with a
Transformer into a context vector that predicts the future features of the other view; contextual contrasting
applies the NT-Xent loss to the projected contexts of the two views. The instance-level representation is the
flattened features, as in the official linear evaluation. The generic heads also give forecasting, classification
and regression; the encoder shortens the sequence, so there is no reconstruction network.

Defaults are those of the official HAR configuration. As in the official code, the weak augmentation only scales
(the paper describes it as jitter-and-scale); unlike the official code, segment permutation moves every channel,
not only the first, and split points fall between steps. The context Transformer is PyTorch's pre-norm encoder
layer, whose attention scales by the head size rather than the model size as in the official code. The projection
head normalizes the contexts of both views in one batch-normalization pass, so a batch of one sequence works.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TSTCCParameters`: encoder, context Transformer, loss weights, augmentations and head. |
| [augmentation.py](./augmentation.py) | `TSTCCAugmentation`: weak scaling and strong permutation-and-jitter augmentations. |
| [sequence_encoder.py](./sequence_encoder.py) | `TSTCCSequenceEncoder`: convolution, batch normalization, ReLU and pooling blocks. |
| [temporal_contrasting.py](./temporal_contrasting.py) | `TSTCCTemporalContrasting`: context Transformer and cross-view prediction loss. |
| [contextual_contrastive_loss.py](./contextual_contrastive_loss.py) | `ContextualContrastiveLoss`: NT-Xent loss of the projected contexts. |
| [pretraining_network.py](./pretraining_network.py) | `TSTCCPretrainingNetwork`: augmentations, both contrasting losses and the projection head. |
