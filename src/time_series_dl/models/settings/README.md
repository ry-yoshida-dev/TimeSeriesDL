# settings

## Overview

Selection of a model and its parameters from plain configuration values, so that a configuration file can describe
any model.

## Components

| Component | Description |
| --------- | ----------- |
| [architecture.py](./architecture.py) | `ModelArchitecture`: model family and its parameters class. |
| [model.py](./model.py) | `ModelSettings`: architecture and parameters from a configuration file. |
| [parameters_reader.py](./parameters_reader.py) | `ParametersReader`: builds a parameters dataclass from plain configuration values. |
