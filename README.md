# TimeSeriesDL

## Overview

TimeSeriesDL (`time_series_dl`) puts 32 deep learning models for time series behind one PyTorch interface per task.
Each task is built from a settings dataclass that takes a model's parameters, so models are swapped by changing one
argument; the type checker rejects a model that does not support the task. Training (optimizer, schedule, early
stopping, device) and saving to files that load without pickle are shared by every task, and so are the point
losses (MSE, MAE, Huber, log-cosh, pinball) of the tasks that predict values; classification trains with
cross-entropy or focal loss.

| Task | Interface | Input → output | Models |
| ---- | --------- | -------------- | ------ |
| Forecasting | `Forecaster` | series → point, quantile, Gaussian, Student's t or Laplace forecasts of the next horizon | all 32 |
| Classification | `SequenceClassifier` | (N, L, C) sequences → class | 24 |
| Regression | `SequenceRegressor` | (N, L, C) sequences → one or more values | 24 |
| Imputation | `Imputer` | sequences with NaN → filled sequences | 24 |
| Anomaly detection | `AnomalyDetector` | series → anomaly score of every time step | 24 |

The models range from MLP, RNN, TCN and convolutional classifiers (FCN, ResNet, InceptionTime) through linear and
MLP forecasters (DLinear, NLinear, TSMixer, TiDE, SegRNN) and Transformers (Transformer, Informer, Autoformer,
FEDformer, PatchTST, iTransformer, TimesNet) to forecasting-specific models (N-BEATS, N-HiTS, DeepAR, Temporal
Fusion Transformer), and state space models from HiPPO memories (HiPPO-RNN, LMU) through deep state space models (S4,
S4D, S5, LRU) to Mamba, Mamba-2, S-Mamba and FiLM. Which model supports which task is listed in
[src/time_series_dl/models/](src/time_series_dl/models/README.md). Classification and regression metrics are the
types of [ClassReg](https://github.com/ry-yoshida-dev/ClassReg), and scaling comes from
[DataScaler](https://github.com/ry-yoshida-dev/DataScaler).

## Installation

```bash
uv sync
```

## Usage

Forecast the next 24 steps of channel 0 from the last 96 steps, with prediction intervals:

```python
from time_series_dl import TrainingSettings
from time_series_dl.models import PatchTSTParameters
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow, QuantileObjective

forecaster = ForecasterSettings(
    model=PatchTSTParameters(),
    window=ForecastWindow(input_length=96, horizon=24),
    objective=QuantileObjective((0.1, 0.5, 0.9)),
    target_channels=(0,),
    training=TrainingSettings(max_epochs=50),
).build()
forecaster.fit(train_series, validation_series)  # arrays of shape (length, channels)
prediction = forecaster.forecast(recent_series)
print(prediction.point, prediction.get_quantile(0.9))
print(forecaster.evaluate(test_series).mae)
```

Save the fitted forecaster and load it back, on another device if needed; every task has the same `save` and `load`:

```python
from pathlib import Path

from time_series_dl import Device
from time_series_dl.tasks.forecasting import Forecaster

forecaster.save(Path("forecaster.pt"))
restored = Forecaster.load(Path("forecaster.pt"), device=Device.CPU)
prediction = restored.forecast(recent_series)
```

Classify sequences, or regress values from them, with the same models:

```python
from time_series_dl import HuberLoss
from time_series_dl.models import InceptionTimeParameters
from time_series_dl.tasks.classification import ClassifierSettings
from time_series_dl.tasks.regression import RegressorSettings

classifier = ClassifierSettings(model=InceptionTimeParameters()).build()
classifier.fit(X_train, y_train, validation=(X_valid, y_valid))  # X: (N, L, C)
print(classifier.evaluate(X_test, y_test).macro_f1)

regressor = RegressorSettings(model=InceptionTimeParameters(), loss=HuberLoss(delta=0.5)).build()
regressor.fit(X_train, targets_train)
```

Build a model from a YAML preset in [src/time_series_dl/config/](src/time_series_dl/config/README.md), for example
with [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler), which is not a dependency of this
package:

```python
from dictconfig_handler import DictConfigHandler
from omegaconf import OmegaConf

from time_series_dl.models import ModelSettings

preset_path = "src/time_series_dl/config/patch_tst.yaml"
model = DictConfigHandler(cfg=OmegaConf.load(preset_path)).build_dataclass(ModelSettings, key="model")
forecaster = ForecasterSettings(
    model=model.build_forecasting_model(), window=ForecastWindow(input_length=96, horizon=24)
).build()
classifier = ClassifierSettings(model=model.build_summary_model()).build()
```

Impute missing values and detect anomalies:

```python
from time_series_dl.models import RNNParameters, TimesNetParameters
from time_series_dl.tasks.anomaly_detection import AnomalyDetectorSettings
from time_series_dl.tasks.imputation import ImputerSettings

imputer = ImputerSettings(model=TimesNetParameters()).build()
imputer.fit(windows_with_nan)
filled = imputer.impute(windows_with_nan)

detector = AnomalyDetectorSettings(model=RNNParameters(), window_length=64).build()
detector.fit(normal_series)
is_anomalous = detector.detect(test_series)
```
