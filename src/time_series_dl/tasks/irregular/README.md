# irregular

## Overview

Tasks on irregularly sampled sequences, whose samples have their own observation times and lengths and miss
channels at any time, such as the ICU records of PhysioNet. Sequences are given as an
`IrregularSequenceCollection` of [data/irregular/](../../data/irregular/README.md), and models build the irregular
networks of [continuous_time/](../../models/architectures/continuous_time/README.md).

## Components

| Component | Description |
| --------- | ----------- |
| [classification/](./classification/README.md) | `IrregularSequenceClassifier`: one class per sequence. |
| [regression/](./regression/README.md) | `IrregularSequenceRegressor`: one or more values per sequence. |
| [interpolation/](./interpolation/README.md) | `IrregularInterpolator`: values at any query times, inside or after the observations. |

## Examples

```python
import numpy as np

from time_series_dl.data.irregular import IrregularSequenceCollection, QueryTimes
from time_series_dl.models.architectures.continuous_time import LatentODEParameters, MTANParameters
from time_series_dl.tasks.irregular.classification import IrregularClassifierSettings
from time_series_dl.tasks.irregular.interpolation import IrregularInterpolatorSettings

# One record per ICU stay: hours since admission (length,) and measurements (length, channels), NaN where unmeasured.
records = IrregularSequenceCollection.from_arrays(times, values)
test_records = IrregularSequenceCollection.from_arrays(test_times, test_values)

classifier = IrregularClassifierSettings(model=MTANParameters(), is_class_balanced=True).build()
classifier.fit(records, mortality)
print(classifier.evaluate(test_records, test_mortality).roc_auc)

interpolator = IrregularInterpolatorSettings(model=LatentODEParameters()).build()
interpolator.fit(records)
hourly = interpolator.predict(test_records, QueryTimes.from_arrays([np.arange(48.0)] * test_records.num_sequences))
```
