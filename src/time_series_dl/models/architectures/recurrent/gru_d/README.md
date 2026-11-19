# gru_d

## Overview

GRU-D (Che et al., "Recurrent neural networks for multivariate time series with missing values", Scientific Reports,
2018): a missing input decays from the channel's last observation towards the channel's mean, and the hidden state
decays, both at trainable rates driven by the time since the last observations; the GRU also reads the mask. Each
step's values are read from the decayed hidden state before the step's update, so every estimate uses only earlier
steps; training adds the mean squared error on the observed inputs to the imputer's loss on the values it hid.
Imputation only.

Differences from the paper and PyPOTS:

- The paper defines GRU-D for classification. It is not offered for classification or regression here, because those
  tasks reject missing values, so the mask would always be full and the decays unused.
- The channel means are trainable, starting at zero, the mean of the observed values after the imputer's standard
  scaling; the paper fixes them to the training means. Before a channel's first observation, its mean is used.
- The GRU reads the imputed inputs and the mask, as in the paper; PyPOTS also feeds it the hidden state.
- Time gaps assume equally spaced steps.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `GRUDParameters`: hidden size. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `GRUDMaskedReconstructionNetwork`: input and hidden decays, the GRU and the readout of each step. |
