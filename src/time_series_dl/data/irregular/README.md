# irregular

## Overview

Irregularly sampled sequences, such as ICU records: each sequence has its own observation times and length, and
each channel may be missing at any time. Sequences are validated, padded into arrays with observed masks, and
scaled with statistics of the observed training values.

## Components

| Component | Description |
| --------- | ----------- |
| [sequence.py](./sequence.py) | `IrregularSequence`: times of shape (length,) and values of shape (length, channels), NaN where unobserved. |
| [collection.py](./collection.py) | `IrregularSequenceCollection`: sequences that share their channels, and their padding. |
| [padded_sequences.py](./padded_sequences.py) | `PaddedIrregularSequences`: padded times, values and lengths, and their tensors. |
| [scaler.py](./scaler.py) | `IrregularScaler`: per-channel scaling fitted on observed values, and times mapped to [0, 1]. |
| [query_times.py](./query_times.py) | `QueryTimes`: times at which the values of each sequence are predicted. |

## Examples

```python
import numpy as np

from time_series_dl.data.irregular import IrregularSequenceCollection

sequences = IrregularSequenceCollection.from_arrays(
    times=[np.array([0.0, 1.5, 7.0]), np.array([0.5, 2.0])],
    values=[np.array([[36.6, np.nan], [37.1, 80.0], [np.nan, 92.0]]), np.array([[36.9, 75.0], [37.0, np.nan]])],
)
padded = sequences.pad()  # times (2, 3), values (2, 3, 2) with NaN where unobserved, lengths [3, 2]
```
