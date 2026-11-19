# adversarial

## Overview

Gradient routing for models trained in adversarial phases (USAD, TranAD) through one optimizer: a term computed
through frozen sub-networks only trains the sub-networks left unfrozen.

## Components

| Component | Description |
| --------- | ----------- |
| [frozen_module.py](./frozen_module.py) | `FrozenModule`: calls a module with detached parameters, so gradients reach only its inputs. |
