# classification

## Overview

Classification of whole sequences with any model that builds a summary network. Probabilities and metrics are the
types of `classreg.classification`.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ClassifierSettings`: model, training, scaling, loss and class balancing. |
| [classifier.py](./classifier.py) | `SequenceClassifier`: `fit`, `predict`, `predict_proba`, `evaluate`, `save` and `load`. |
| [classification_step.py](./classification_step.py) | `ClassificationStep`: classification loss of a batch. |
