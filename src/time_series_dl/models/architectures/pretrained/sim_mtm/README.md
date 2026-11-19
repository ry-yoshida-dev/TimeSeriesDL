# sim_mtm

## Overview

SimMTM (Dong et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", NeurIPS 2023):
every sequence is masked several times; the encoder of any model encodes the originals and the masked versions,
a projector turns them into series-wise representations, and each original is reconstructed from the point-wise
features of all other sequences aggregated by the softmax of their series-wise similarities. A constraint loss
makes the versions of a sequence each other's most similar neighbors.

Unlike the official code, the encoder reads the channels of a sequence together rather than each channel on its
own, sequences are scaled by the task rather than normalized one by one, the series-wise representation averages
the features over time by default instead of flattening them, and only the originals are decoded (the loss uses no
other decoded sequence). A fixed constraint weight, as in the paper, may replace the learned loss weights.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `SimMTMParameters`: encoder, masking, temperature, projection and loss weighting. |
| [pretraining_network.py](./pretraining_network.py) | `SimMTMPretrainingNetwork`: masked versions, constraint loss, aggregation and reconstruction. |
| [automatic_loss_weighting.py](./automatic_loss_weighting.py) | `SimMTMAutomaticLossWeighting`: learned weights of the two losses. |
