# ts2vec

## Overview

TS2Vec (Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022): a linear projection of
each step, timestamp masking and residual blocks of dilated convolutions encode every timestamp; pretraining
contrasts two overlapping random crops of each sequence on their overlap, instance-wise and over time, at every
level of a max-pooling hierarchy. The instance-level representation is the maximum over time. The generic heads
also give every supervised task, and `PretrainedEncoderParameters` transfers a pretrained encoder.

Unlike the paper, sequences have a fixed length and no missing values, so there is no NaN handling or padding, and
long sequences are not split into `max_train_length` pieces. Representations of the whole sequence are computed in
one pass: the causal sliding encoding of the paper's forecasting and anomaly detection protocols and the multiscale
encoding windows are not provided. Timestamps are masked only during pretraining, never in the networks of other
tasks.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TS2VecParameters`: encoder sizes, masking, loss weighting and head. |
| [convolution_block.py](./convolution_block.py) | `TS2VecConvolutionBlock`: residual pair of dilated convolutions. |
| [sequence_encoder.py](./sequence_encoder.py) | `TS2VecSequenceEncoder`: projection, masking and dilated convolutions. |
| [hierarchical_contrastive_loss.py](./hierarchical_contrastive_loss.py) | `HierarchicalContrastiveLoss`: instance-wise and temporal contrast at every level. |
| [pretraining_network.py](./pretraining_network.py) | `TS2VecPretrainingNetwork`: random crops, masks and the loss of their overlap. |
