# native

## Overview

Native scoring: models that build an `AnomalyScoringNetwork`, such as the Anomaly Transformer, DCdetector, TranAD
and USAD, train with their own objective and score with their own criterion.

## Components

| Component | Description |
| --------- | ----------- |
| [anomaly_scoring.py](./anomaly_scoring.py) | `NativeScoring`: builds the model's anomaly scoring network. |
