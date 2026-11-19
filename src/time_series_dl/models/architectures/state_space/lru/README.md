# lru

## Overview

Linear Recurrent Unit (Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences", ICML 2023): a
complex diagonal linear recurrence parametrized directly in discrete time by the logs of its rate and phase, with
eigenvalues initialized on a ring, input normalization, and a parallel scan. The blocks use layer normalization and
the GELU and gated linear unit output stage.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `LRUParameters`: sizes, ring radii, maximum phase, layers, dropout, direction and head. |
