# usad

## Overview

USAD (Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate Time Series", KDD 2020): two
autoencoders with a shared encoder over the flattened window, trained in two adversarial phases whose weights shift
from reconstruction (1/n) to adversarial (1 - 1/n) over the epochs n. The anomaly score weighs the first
autoencoder's reconstruction error by alpha and the second's reconstruction of the first's output by beta.

Differences from the paper and the official code:

- The two optimizers of the paper are replaced by one loss whose gradient is the sum of both phase gradients at the
  current weights (see `USADAnomalyScoringNetwork`); the official code steps the two optimizers one after the other.
- The loss value the trainer monitors is the unweighted sum of the two reconstruction errors; its gradient is the
  paper's.
- Scores are per time step (averaged over channels and over the windows covering the step) instead of one score per
  window.

The decoders end with a sigmoid as in the paper, so scale the channels to [0, 1] as the paper does, with
`scaling=MinMaxScalingParameters()` in `AnomalyDetectorSettings`.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `USADParameters`: latent size and the alpha/beta score weights. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `USADAnomalyScoringNetwork`: shared encoder, two decoders, combined adversarial loss and score. |
