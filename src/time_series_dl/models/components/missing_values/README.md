# missing_values

## Overview

Building blocks of the recurrent imputation models that read the observed mask, such as GRU-D and BRITS.

## Components

| Component | Description |
| --------- | ----------- |
| [time_gap.py](./time_gap.py) | `TimeGap`: time since the previous observation of each channel. |
| [temporal_decay.py](./temporal_decay.py) | `TemporalDecay`: decay rates in (0, 1] learned from the time gaps. |
| [last_observation.py](./last_observation.py) | `LastObservation`: last observation carried forward. |
