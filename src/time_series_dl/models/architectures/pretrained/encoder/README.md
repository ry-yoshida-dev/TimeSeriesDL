# encoder

## Overview

Transfer of a pretrained encoder: `PretrainedEncoderParameters` reads the encoder of a file written by
`RepresentationLearner.save` and puts the generic heads of every task around it, so a classifier, regressor,
forecaster or imputer starts from the pretrained weights. The encoder is fine-tuned with the head, or frozen for
linear probing. In a YAML preset or `ModelSettings`, the model is given by the path of that file
(`checkpoint_path`), which must still exist whenever a network is built, including when a fitted task is loaded.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `PretrainedEncoderParameters`: file path, freezing and head. |
| [frozen_sequence_encoder.py](./frozen_sequence_encoder.py) | `FrozenSequenceEncoder`: an encoder without gradients, always in evaluation mode. |

## Examples

```python
from time_series_dl.models import PretrainedEncoderParameters
from time_series_dl.tasks.classification import ClassifierSettings

probe = ClassifierSettings(
    model=PretrainedEncoderParameters(checkpoint_path="ts2vec.pt", is_encoder_frozen=True)
).build()
probe.fit(X_train, y_train)
```
