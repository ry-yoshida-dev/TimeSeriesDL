# TimeSeriesDL

## Overview

TimeSeriesDL (`time_series_dl`) puts 64 deep learning models for time series behind one PyTorch interface per task.
Each task is built from a settings dataclass that takes a model's parameters, so models are swapped by changing one
argument; the type checker rejects a model that does not support the task. Training (optimizer, schedule, early
stopping, device) and saving to files that load without pickle are shared by every task, and so are the point
losses (MSE, MAE, Huber, log-cosh, pinball) of the tasks that predict values; classification trains with
cross-entropy or focal loss.

| Task | Interface | Input → output | Models |
| ---- | --------- | -------------- | ------ |
| Forecasting | `Forecaster` | series → point, quantile, Gaussian, Student's t or Laplace forecasts of the next horizon | 58 |
| Classification | `SequenceClassifier` | (N, L, C) sequences → class | 42 |
| Regression | `SequenceRegressor` | (N, L, C) sequences → one or more values | 42 |
| Imputation | `Imputer` | sequences with NaN → filled sequences | 42 |
| Anomaly detection | `AnomalyDetector` | series → anomaly score of every time step | 39 |

The models span linear and MLP forecasters (DLinear, RLinear, FITS, SparseTSF, TSMixer, TiDE, TimeMixer,
TimeMixer++, FreTS, Koopa, N-BEATS, N-HiTS), recurrent networks (LSTM/GRU, SegRNN, DeepAR, xLSTM, xLSTMTime,
minGRU/minLSTM, LSTNet), convolutional networks and classifiers (FCN, ResNet, InceptionTime, LITE, TCN, TimesNet,
MICN, ModernTCN, ROCKET, MiniRocket, MultiRocket), Transformers (Transformer, Informer, Autoformer, FEDformer,
Pyraformer, ETSformer, Non-stationary Transformer, Crossformer, PatchTST, iTransformer, TimeXer, TST, ConvTran,
Temporal Fusion Transformer), state space models (HiPPO-RNN, LMU, S4, S4D, S5, LRU, Mamba, Mamba-2, S-Mamba, FiLM)
graph neural networks over the channels (MTGNN, Graph WaveNet, AGCRN, StemGNN, FourierGNN), and imputation models
that read the missing-value mask (SAITS, BRITS, GRU-D). Forecasts accept
known-future and static covariates with every forecasting model. Which model supports which task is listed in
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

Add known-future covariates (calendar features, planned promotions) and static features of each series; the known
covariates passed to `forecast` extend `horizon` steps past the series. TiDE, DeepAR and the Temporal Fusion
Transformer read them natively, and every other forecasting model through a generic adapter:

```python
import numpy as np

from time_series_dl.models import TiDEParameters
from time_series_dl.tasks.forecasting import CalendarFeatures, ForecasterSettings, ForecastWindow, SeriesCovariates

known = np.concatenate([CalendarFeatures().compute(timestamps), promotions], axis=1)  # (length + 24, 5)
forecaster = ForecasterSettings(model=TiDEParameters(), window=ForecastWindow(input_length=96, horizon=24)).build()
forecaster.fit(sales, covariates=SeriesCovariates(known=known[:-24], static_categories=np.array([store_id])))
prediction = forecaster.forecast(sales, SeriesCovariates(known=known, static_categories=np.array([store_id])))
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
from time_series_dl import MAELoss
from time_series_dl.models import RNNParameters, SAITSParameters
from time_series_dl.tasks.anomaly_detection import AnomalyDetectorSettings
from time_series_dl.tasks.imputation import ImputerSettings

imputer = ImputerSettings(model=SAITSParameters(), loss=MAELoss(), mask_ratio=0.2).build()
imputer.fit(windows_with_nan)
filled = imputer.impute(windows_with_nan)

detector = AnomalyDetectorSettings(model=RNNParameters(), window_length=64).build()
detector.fit(normal_series)
is_anomalous = detector.detect(test_series)
```
