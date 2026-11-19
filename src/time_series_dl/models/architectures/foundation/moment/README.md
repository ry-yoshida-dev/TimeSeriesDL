# moment

## Overview

MOMENT (Goswami et al., "MOMENT: A Family of Open Time-series Foundation Models", ICML 2024): each channel is a
separate series of a fixed 512 steps, normalized by the mean and standard deviation of its observed values (RevIN
without affine parameters), cut into non-overlapping patches of 8 values, embedded linearly with a learned mask
embedding for hidden patches and a sinusoidal position embedding, and encoded by a Flan-T5 encoder pretrained to
reconstruct masked patches. `MOMENTParameters.pretrained(MOMENTSize.SMALL | BASE | LARGE)` loads the published
weights of `AutonLab/MOMENT-1-*` (MIT license) at a pinned revision; the defaults are the MOMENT-1-large architecture
without weights.

Only the reconstruction head is pretrained. Reconstruction networks (anomaly detection, with nothing hidden as in the
official anomaly detection) and masked reconstruction networks (imputation: a missing value excludes its patch from
the input, replaced by the mask embedding, and from the normalization statistics) therefore work zero-shot with
published weights. Forecast networks (flattened patch states to the horizon) and summary networks (per-channel mean
of the patch states, concatenated over channels, to the output vector) get a new head and must be fine-tuned; as in
the official fine-tuning, `is_backbone_frozen` (set by `pretrained`) trains only the head. MOMENT normalizes each series itself,
so the tasks' scaling and the forecaster's instance normalization are best left off (`NoScalingParameters()`,
`is_instance_normalized=False`), although its own normalization makes them harmless.

Differences from the official code: each channel has its own missing-value mask instead of one shared by the
channels of a sample (identical when channels are missing together); a series without any observed value gets
location 0 and scale 1 instead of NaN statistics; series shorter than 512 steps are left-padded by the networks
(the official code expects callers to pad), and as in the official default a patch that is partly padding is hidden
and excluded from the statistics; forecast windows longer than 512 steps are cut to their last 512 steps, while
reconstruction and summary networks reject sequences longer than 512 steps; forecast networks output
`num_parameters` values per step, of which only those in the scale of the values are denormalized; the classification
head's `mean` reduction, prompt tuning and gradient checkpointing are not provided.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MOMENTParameters`: encoder sizes, patching, dropout, freezing and published weights. |
| [size.py](./size.py) | `MOMENTSize`: published checkpoints and their pinned revisions. |
| [patch_embedding.py](./patch_embedding.py) | `MOMENTPatchEmbedding`: linear patch embedding, mask embedding and sinusoidal positions. |
| [encoding.py](./encoding.py) | `MOMENTEncoding`: patch states and the normalization statistics. |
| [encoder.py](./encoder.py) | `MOMENTEncoder`: masked instance normalization, patching, embedding and the T5 encoder. |
| [reconstruction_head.py](./reconstruction_head.py) | `MOMENTReconstructionHead`: the pretrained head from patch states to values. |
| [reconstruction_network.py](./reconstruction_network.py) | `MOMENTReconstructionNetwork`: zero-shot reconstruction with nothing hidden. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `MOMENTMaskedReconstructionNetwork`: zero-shot reconstruction with missing values hidden. |
| [forecast_network.py](./forecast_network.py) | `MOMENTForecastNetwork`: flattened patch states of each target channel to its horizon. |
| [summary_network.py](./summary_network.py) | `MOMENTSummaryNetwork`: mean patch states of all channels to the output vector. |
| [official_names.py](./official_names.py) | `MOMENTOfficialNames`: renamings of the published parameter names. |

## Examples

```python
from data_scaler import NoScalingParameters

from time_series_dl.models.architectures.foundation.moment import MOMENTParameters, MOMENTSize
from time_series_dl.tasks.anomaly_detection import AnomalyDetectorSettings

model = MOMENTParameters.pretrained(MOMENTSize.SMALL)  # downloads AutonLab/MOMENT-1-small when a network is built
detector = AnomalyDetectorSettings(model=model, window_length=512, scaling=NoScalingParameters()).build()
```
