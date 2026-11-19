# maskings

## Overview

Implementations of `Masking`, the random choice of the values that masked modeling hides.

## Components

| Component | Description |
| --------- | ----------- |
| [geometric.py](./geometric.py) | `GeometricMasking`: hidden segments of geometric length in each channel (TST, SimMTM). |
| [patch.py](./patch.py) | `PatchMasking`: a fixed fraction of the non-overlapping patches of each channel (PatchTST). |
