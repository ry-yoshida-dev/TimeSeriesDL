# ode_rnn

## Overview

ODE-RNN (Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019). Between observations the
hidden state follows a neural ordinary differential equation; at each observation a GRU cell updates it from the
values and the observed mask. The state starts at zero at time 0 (the earliest training time), and its value at a
sequence's last observation feeds the output layers. The same recurrence, read backward, is the recognition network
of the Latent ODE.

Differences from the paper: the update is a standard GRU cell instead of the authors' GRU that also tracks a
standard deviation of the state; the equation is solved with `steps_per_interval` fixed steps per observation
interval instead of a step count proportional to the interval; observations are processed per sample rather than on
the union of the batch's times.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ODERNNParameters`: sizes, solver and head. |
| [recurrence.py](./recurrence.py) | `ODERNNRecurrence`: the ODE-RNN, read forward or backward. |
| [irregular_sequence_encoder.py](./irregular_sequence_encoder.py) | `ODERNNSequenceEncoder`: the forward ODE-RNN as an irregular sequence encoder. |
