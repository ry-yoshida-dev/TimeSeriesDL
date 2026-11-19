# pretrained

## Overview

Loading of officially published weights into this package's networks. A `PretrainedWeights` names a safetensors
file in a Hugging Face Hub repository at a pinned revision; `PretrainedWeightsLoader` downloads it into the local
cache, renames its parameters with the model's `ParameterNameMapping` and loads them strictly: a missing,
unexpected or differently shaped tensor raises `WeightsMismatchError`. Downloading and reading need the optional
`pretrained` extra (`huggingface-hub`, `safetensors`); without it `MissingDependencyError` is raised. Requests go
only to the Hugging Face Hub, with a fixed `time-series-dl` user agent in place of huggingface_hub's default, which
reports the installed torch version and any detected agent harness. `HF_HUB_OFFLINE=1` reads only the cache.

## Components

| Component | Description |
| --------- | ----------- |
| [weights.py](./weights.py) | `PretrainedWeights`: repository, revision and file of published weights. |
| [hub_client.py](./hub_client.py) | `HubClient`: downloads a weight file into the Hugging Face cache. |
| [safetensors_file.py](./safetensors_file.py) | `SafetensorsFile`: reads and writes safetensors files. |
| [name_rule.py](./name_rule.py) | `NameRule`: one regular-expression renaming of published parameter names. |
| [name_mapping.py](./name_mapping.py) | `ParameterNameMapping`: ordered renamings and ignored published names. |
| [weights_loader.py](./weights_loader.py) | `PretrainedWeightsLoader`: strict loading of published weights into a network. |
| [weights_mismatch_error.py](./weights_mismatch_error.py) | `WeightsMismatchError`: missing, unexpected and mismatched tensors. |
| [missing_dependency_error.py](./missing_dependency_error.py) | `MissingDependencyError`: the `pretrained` extra is not installed. |

## Examples

```python
from time_series_dl.models.architectures.foundation.chronos_bolt import ChronosBoltParameters, ChronosBoltSize

# Downloads amazon/chronos-bolt-small at a pinned revision when the network is built.
parameters = ChronosBoltParameters.pretrained(ChronosBoltSize.SMALL)
```
