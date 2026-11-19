# neural_cde

## Overview

Neural CDE (Kidger et al., "Neural Controlled Differential Equations for Irregular Time Series", NeurIPS 2020). The
hidden state follows `dz = f(z) dX`, where `X` is a cubic interpolation of the time, the values filled forward in
each channel and the observational intensity (cumulative count of observations) of each channel; the final hidden
state feeds a linear readout.

Differences from the paper: the control path is the Hermite cubic with backward differences that the authors
recommend later (Morrill et al., 2021) by default, with the natural cubic spline of the paper as an option; the
equation is solved with a fixed-step solver, `steps_per_interval` steps per observation interval over the position
within the interval (equivalent, since the equation is invariant to reparametrizing time), instead of the adaptive
solvers of `torchdiffeq`; values before a channel's first observation are 0 (the scaled mean); the observational
intensity is the raw count, which enters only through its increments. Sizes approximate the paper's PhysioNet
sepsis experiment with observational intensity.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `NeuralCDEParameters`: sizes, interpolation, solver and head. |
| [vector_field.py](./vector_field.py) | `NeuralCDEVectorField`: hidden state to the matrix that multiplies the path's derivative. |
| [irregular_sequence_encoder.py](./irregular_sequence_encoder.py) | `NeuralCDESequenceEncoder`: the control path and the solved equation. |
