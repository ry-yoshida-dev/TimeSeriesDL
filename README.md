# TimeSeriesDL

## Overview

TimeSeriesDL (`time_series_dl`) puts 84 deep learning models for time series behind one PyTorch interface per task.
Each task is built from a settings dataclass that takes a model's parameters, so models are swapped by changing one
argument; the type checker rejects a model that does not support the task. Training (optimizer, schedule, early
stopping, device) and saving to files that load without pickle are shared by every task. Losses come from
[DeepLearningLoss](https://github.com/ry-yoshida-dev/DeepLearningLoss): any `PointLoss` (MSE, MAE, Huber, log-cosh,
Charbonnier, pinball, MAPE, sMAPE, MSLE) trains the tasks that predict values, and any `ClassificationLoss`
(cross-entropy, focal, generalized cross-entropy, hinge, Poly-1) trains classification.

| Task | Interface | Input → output | Models |
| ---- | --------- | -------------- | ------ |
| Forecasting | `Forecaster` | series → point, quantile, parametric, multivariate low-rank Gaussian or sample-path forecasts of the next horizon, with conformal intervals | 70 |
| Classification | `SequenceClassifier` | (N, L, C) sequences → class | 46 |
| Regression | `SequenceRegressor` | (N, L, C) sequences → one or more values | 46 |
| Imputation | `Imputer` | sequences with NaN → filled sequences | 46 |
| Anomaly detection | `AnomalyDetector` | series → anomaly score of every time step | 70 |
| Irregular classification | `IrregularSequenceClassifier` | irregularly sampled sequences → class | 4 |
| Irregular regression | `IrregularSequenceRegressor` | irregularly sampled sequences → one or more values | 4 |
| Irregular interpolation | `IrregularInterpolator` | irregularly sampled sequences → values at query times | 2 |
| Representation learning | `RepresentationLearner` | unlabeled sequences → pretrained encoder and representations | 4 |

The models span linear and MLP forecasters (DLinear, RLinear, FITS, SparseTSF, TSMixer, TiDE, TimeMixer,
TimeMixer++, FreTS, Koopa, N-BEATS, N-HiTS), recurrent networks (LSTM/GRU, SegRNN, DeepAR, xLSTM, xLSTMTime,
minGRU/minLSTM, LSTNet), convolutional networks and classifiers (FCN, ResNet, InceptionTime, LITE, TCN, TimesNet,
MICN, ModernTCN, ROCKET, MiniRocket, MultiRocket), Transformers (Transformer, Informer, Autoformer, FEDformer,
Pyraformer, ETSformer, Non-stationary Transformer, Crossformer, PatchTST, iTransformer, TimeXer, TST, ConvTran,
Temporal Fusion Transformer), state space models (HiPPO-RNN, LMU, S4, S4D, S5, LRU, Mamba, Mamba-2, S-Mamba, FiLM)
graph neural networks over the channels (MTGNN, Graph WaveNet, AGCRN, StemGNN, FourierGNN), imputation models that read the
missing-value mask (SAITS, BRITS, GRU-D), anomaly detectors with their own objectives (Anomaly Transformer,
DCdetector, TranAD, USAD), self-supervised pretraining (TS2Vec, TS-TCC, SimMTM, masked
reconstruction) whose encoders transfer to the other tasks, and continuous-time models of irregularly sampled
sequences (mTAN, Neural CDE, ODE-RNN, Latent ODE). Foundation models (Chronos,
Chronos-Bolt, TimesFM, Moirai, MOMENT, Timer, TTM) load their published weights to forecast zero-shot or to be
fine-tuned. Generative forecasters (TimeGrad, CSDI, Chronos, Moirai) draw sample paths, and conformal prediction calibrates the intervals of any forecaster. Forecasts
accept known-future and static covariates with every forecasting model. Which model supports which task is listed in
[src/time_series_dl/models/](src/time_series_dl/models/README.md). Classification and regression metrics are the
types of [ClassReg](https://github.com/ry-yoshida-dev/ClassReg), and scaling comes from
[DataScaler](https://github.com/ry-yoshida-dev/DataScaler).

## Installation

```bash
uv sync
```

The foundation models download their published weights from the Hugging Face Hub through an optional extra:

```bash
uv sync --extra pretrained  # or: pip install "time-series-dl[pretrained]"
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

Draw sample paths with a diffusion model, measure them by CRPS, and calibrate 90% intervals on held-out series with
conformal prediction:

```python
from time_series_dl.models import CSDIParameters
from time_series_dl.tasks.forecasting import ConformalSettings, ForecasterSettings, ForecastWindow, SampleObjective

forecaster = ForecasterSettings(
    model=CSDIParameters(), window=ForecastWindow(input_length=96, horizon=24), objective=SampleObjective()
).build()
forecaster.fit(train_series)
prediction = forecaster.forecast(recent_series, num_samples=100)  # prediction.samples: (1, 100, 24, channels)
print(forecaster.evaluate(test_series).crps)
conformal = ConformalSettings(coverage_levels=(0.9,)).build(forecaster)
conformal.calibrate(calibration_series)
interval = conformal.forecast(recent_series)  # interval.get_quantile(0.05), interval.get_quantile(0.95)
```

Forecast zero-shot with a foundation model's published weights, then fine-tune it; these models scale each window
themselves, so the task's scaling is turned off:

```python
from data_scaler import NoScalingParameters

from time_series_dl.models import ChronosBoltParameters, ChronosBoltSize
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow, QuantileObjective

model = ChronosBoltParameters.pretrained(ChronosBoltSize.SMALL)
forecaster = ForecasterSettings(
    model=model,
    window=ForecastWindow(input_length=512, horizon=64),
    objective=QuantileObjective(model.quantile_levels),
    scaling=NoScalingParameters(),
    is_instance_normalized=False,
).build()
forecaster.prepare_zero_shot(series)  # downloads the weights once; no training
prediction = forecaster.forecast(series)
forecaster.fit(train_series)  # fine-tunes from the published weights
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
from deep_learning_loss import HuberLoss
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
from deep_learning_loss import MAELoss
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

Detect anomalies with a model's own objective, or score any forecasting model by its forecast errors; metrics include
range-based F1 and, for comparison with papers only, point-adjusted F1:

```python
from data_scaler import MinMaxScalingParameters

from time_series_dl.models import DLinearParameters, USADParameters
from time_series_dl.tasks.anomaly_detection import AnomalyDetectorSettings, ForecastErrorScoring

detector = AnomalyDetectorSettings(model=USADParameters(), window_length=12, scaling=MinMaxScalingParameters()).build()
detector.fit(normal_series)
metrics = detector.evaluate(test_series, labels)
print(metrics.f1, metrics.range_f1, metrics.point_adjusted_f1)
forecast_detector = AnomalyDetectorSettings(
    model=DLinearParameters(), window_length=96, scoring=ForecastErrorScoring()
).build()
```

Pretrain an encoder on unlabeled sequences, then train a linear probe on its frozen representations:

```python
from pathlib import Path

from time_series_dl.models import PretrainedEncoderParameters, TS2VecParameters
from time_series_dl.tasks.classification import ClassifierSettings
from time_series_dl.tasks.representation_learning import RepresentationLearnerSettings

learner = RepresentationLearnerSettings(method=TS2VecParameters()).build()
learner.fit(unlabeled)  # (N, L, C) sequences without labels
representations = learner.encode(X_train)  # (N, 320)
learner.save(Path("ts2vec.pt"))

probe = ClassifierSettings(
    model=PretrainedEncoderParameters(checkpoint_path="ts2vec.pt", is_encoder_frozen=True)
).build()
probe.fit(X_train, y_train)
```

Classify irregularly sampled records, such as ICU stays, and interpolate them at query times:

```python
import numpy as np

from time_series_dl.data import IrregularSequenceCollection, QueryTimes
from time_series_dl.models import LatentODEParameters, MTANParameters
from time_series_dl.tasks.irregular.classification import IrregularClassifierSettings
from time_series_dl.tasks.irregular.interpolation import IrregularInterpolatorSettings

# times: hours since admission per record; values: (length, channels) per record, NaN where unmeasured.
records = IrregularSequenceCollection.from_arrays(times, values)
classifier = IrregularClassifierSettings(model=MTANParameters(), is_class_balanced=True).build()
classifier.fit(records, mortality)

interpolator = IrregularInterpolatorSettings(model=LatentODEParameters()).build()
interpolator.fit(records)
hourly = interpolator.predict(records, QueryTimes.from_arrays([np.arange(48.0)] * records.num_sequences))
```
