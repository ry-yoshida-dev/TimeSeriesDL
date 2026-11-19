# classification

## Overview

Classification of irregularly sampled sequences, such as in-hospital mortality from PhysioNet ICU records, with any
model that builds an irregular summary network. Losses are those of [classification](../../classification/README.md),
and probabilities and metrics are the types of `classreg.classification`.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `IrregularClassifierSettings`: model, training, scaling, loss and class balancing. |
| [classifier.py](./classifier.py) | `IrregularSequenceClassifier`: `fit`, `predict`, `predict_proba`, `evaluate`, `save` and `load`. |
| [classification_step.py](./classification_step.py) | `IrregularClassificationStep`: training loss of a batch, with the network's own terms. |
