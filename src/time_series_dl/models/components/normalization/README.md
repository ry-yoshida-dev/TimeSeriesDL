# normalization

## Overview

Normalization layers that are part of a model's architecture, as opposed to the instance normalization the
forecasting task applies around every network.

## Components

| Component | Description |
| --------- | ----------- |
| [reversible_instance_normalization.py](./reversible_instance_normalization.py) | `ReversibleInstanceNormalization`: RevIN with a learnable per-channel affine transformation. |
| [instance_statistics.py](./instance_statistics.py) | `InstanceStatistics`: the location and scale of each window and channel. |
