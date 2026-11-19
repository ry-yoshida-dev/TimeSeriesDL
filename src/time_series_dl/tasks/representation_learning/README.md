# representation_learning

## Overview

Self-supervised pretraining of an encoder on unlabeled sequences or series windows with any model that builds a
pretraining network (TS2Vec, TS-TCC, SimMTM, masked reconstruction), and the instance-level and timestamp-level
representations of new sequences. `PretrainedEncoderParameters` transfers the saved encoder to classification,
regression, forecasting and imputation, fine-tuned or frozen.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `RepresentationLearnerSettings`: pretraining method, training and scaling. |
| [learner.py](./learner.py) | `RepresentationLearner`: `fit`, `fit_series`, `encode`, `encode_timestamps`, `save` and `load`. |
| [pretraining_step.py](./pretraining_step.py) | `PretrainingStep`: self-supervised loss of a batch. |

## Examples

```python
from pathlib import Path

from time_series_dl.models import PretrainedEncoderParameters, TS2VecParameters
from time_series_dl.tasks.classification import ClassifierSettings
from time_series_dl.tasks.representation_learning import RepresentationLearnerSettings

learner = RepresentationLearnerSettings(method=TS2VecParameters()).build()
learner.fit(unlabeled)  # (N, L, C)
representations = learner.encode(X_train)  # (N, 320)
learner.save(Path("ts2vec.pt"))

probe = ClassifierSettings(
    model=PretrainedEncoderParameters(checkpoint_path="ts2vec.pt", is_encoder_frozen=True)
).build()
probe.fit(X_train, y_train)
```
