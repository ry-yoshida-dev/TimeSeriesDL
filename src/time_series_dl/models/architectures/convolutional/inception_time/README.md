# inception_time

## Overview

InceptionTime (Ismail Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification", Data Mining and
Knowledge Discovery 2020): one network of residual Inception modules as a sequence encoder, with the generic heads of
every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `InceptionTimeParameters`: filters, bottleneck, kernel lengths, depth, shortcuts and head. |
| [inception_module.py](./inception_module.py) | `InceptionTimeModule`: parallel convolutions of several lengths and a max-pooling branch. |
| [sequence_encoder.py](./sequence_encoder.py) | `InceptionTimeSequenceEncoder`: the stack of modules with shortcuts. |
