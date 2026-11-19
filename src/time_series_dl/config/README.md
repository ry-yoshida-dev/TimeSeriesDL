# config

## Overview

Ready-made YAML presets of every model, one file per `ModelArchitecture` value, holding the defaults of its
parameters class (the paper's settings where the model's README gives them). Each preset's `model` section maps onto
the fields of [`ModelSettings`](../models/settings/model.py); copy a preset into an application config and edit the
parameters, or leave a parameter out to keep its default.

## Examples

```yaml
model:
  architecture: patch_tst
  parameters:
    patch_length: 16
    stride: 8
    model_size: 128
    num_heads: 16
    num_layers: 3
    feedforward_size: 256
    dropout: 0.2
    head_dropout: 0.0
    activation: gelu
    normalization: batch
```
