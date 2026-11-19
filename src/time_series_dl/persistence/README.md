# persistence

## Overview

The file format behind the `save` and `load` methods of every task. A checkpoint holds the settings, the fitted
scalers of the series and of any covariates, the network weights and the training history in one file that
`torch.load` reads with `weights_only=True`, so loading an untrusted file cannot run code.

## Components

| Component | Description |
| --------- | ----------- |
| [checkpoint.py](./checkpoint.py) | `Checkpoint`: the contents of a saved task and its file I/O. |
| [settings_codec.py](./settings_codec.py) | `SettingsCodec`: settings dataclasses to and from plain values, keeping the implementation of each model, loss and objective. |
| [task_kind.py](./task_kind.py) | `TaskKind`: the task a checkpoint belongs to. |
| [encoded_value.py](./encoded_value.py) | `EncodedValue`: the plain values a checkpoint may hold. |
| [restorable_model.py](./restorable_model.py) | `restorable_model`: the parameters that rebuild a saved network without downloading its published weights. |
