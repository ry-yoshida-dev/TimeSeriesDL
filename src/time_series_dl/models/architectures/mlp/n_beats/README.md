# n_beats

## Overview

N-BEATS (Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting",
ICLR 2020): doubly residual stacks of fully connected blocks that expand coefficients with generic, polynomial trend
or Fourier seasonality bases. Each target channel is forecast from its own history. The block and the stacking are
shared with N-HiTS.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `NBEATSParameters`: stacks, trend degree and harmonics, with generic and interpretable presets. |
| [stack.py](./stack.py) | `NBEATSStack`: basis, blocks, layers, width and weight sharing of one stack. |
| [block_kind.py](./block_kind.py) | `NBEATSBlockKind`: generic, trend or seasonality basis. |
| [basis_expansion_block.py](./basis_expansion_block.py) | `BasisExpansionBlock`: backcast and forecast of one block. |
| [doubly_residual_stacking.py](./doubly_residual_stacking.py) | `DoublyResidualStacking`: residual chaining of blocks over each channel. |
| [forecast_network.py](./forecast_network.py) | `NBEATSForecastNetwork`: forecasts the target channels. |

## Examples

```python
from time_series_dl.models.n_beats import NBEATSParameters

generic = NBEATSParameters.generic(num_stacks=6, hidden_size=256)
interpretable = NBEATSParameters.interpretable()
```
