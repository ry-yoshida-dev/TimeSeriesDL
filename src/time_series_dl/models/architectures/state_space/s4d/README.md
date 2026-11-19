# s4d

## Overview

S4D (Gu et al., "On the Parameterization and Initialization of Diagonal State Space Models", NeurIPS 2022): S4 with
complex diagonal state matrices, initialized by S4D-LegS, S4D-Inv or S4D-Lin and discretized by zero-order hold or
the bilinear transform. The input matrix is fixed to ones, as in the paper's minimal version, and the state space
parameters use the optimizer's learning rate and weight decay.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `S4DParameters`: sizes, initialization, discretization, step sizes, layers, dropout, direction and head. |
