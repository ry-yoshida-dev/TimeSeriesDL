# brits

## Overview

BRITS (Cao et al., "BRITS: Bidirectional recurrent imputation for time series", NeurIPS 2018): in each direction, a
recurrent imputation (RITS) estimates every step from the decayed hidden state and, by feature regression, from the
other channels of the step, combines the two, and feeds the completed step to an LSTM cell. The forward and backward
estimates are averaged. Training uses the network's own loss: the mean absolute error of the three estimates on the
observed inputs in both directions, plus the consistency loss between the directions. Imputation only.

Differences from the paper and PyPOTS:

- Time gaps assume equally spaced steps.
- The imputer's loss on the values it hid is not trained on (weight zero), as in the paper; those values are still
  hidden from the inputs while training, and validation measures them.
- The estimation losses are averaged over the steps and the three estimates, as in PyPOTS.
- The joint classification of the paper is not offered.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `BRITSParameters`: hidden size and consistency weight. |
| [feature_regression.py](./feature_regression.py) | `BRITSFeatureRegression`: each channel estimated from the other channels. |
| [recurrent_imputation.py](./recurrent_imputation.py) | `BRITSRecurrentImputation`: RITS, the imputation of one direction, and its estimation loss. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `BRITSMaskedReconstructionNetwork`: both directions, their average and the consistency loss. |
