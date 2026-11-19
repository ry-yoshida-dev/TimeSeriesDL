# runtime

## Overview

Choice of the device that networks are trained and run on.

## Components

| Component | Description |
| --------- | ----------- |
| [device.py](./device.py) | `Device`: automatic, CPU, CUDA or MPS, resolved to a `torch.device`. |
| [batched_inference.py](./batched_inference.py) | `BatchedInference`: runs a function of batches over a dataset without gradients. |
