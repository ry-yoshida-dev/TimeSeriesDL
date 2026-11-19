# reconstruction_error

## Overview

Reconstruction-error scoring: a reconstruction network trained on normal windows, scoring each step by its
reconstruction loss averaged over channels.

## Components

| Component | Description |
| --------- | ----------- |
| [anomaly_scoring.py](./anomaly_scoring.py) | `ReconstructionErrorScoring`: builds the reconstruction network and its adapter. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `ReconstructionErrorScoringNetwork`: reconstruction loss for training and for the scores. |
