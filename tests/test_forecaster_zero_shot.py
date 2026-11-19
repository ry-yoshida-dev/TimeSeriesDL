from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from data_scaler import NoScalingParameters
from test_model_chronos_bolt import TINY, published_state_dict
from torch import Tensor

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import MLPParameters
from time_series_dl.models.architectures.foundation.chronos import ChronosParameters
from time_series_dl.models.architectures.foundation.chronos_bolt import (
    ChronosBoltForecastNetwork,
    ChronosBoltParameters,
)
from time_series_dl.models.components.pretrained import HubClient, PretrainedWeights, SafetensorsFile
from time_series_dl.models.interfaces import ForecastingModel, ForecastShape
from time_series_dl.tasks.forecasting import (
    Forecaster,
    ForecasterSettings,
    ForecastObjective,
    ForecastWindow,
    GaussianObjective,
    PointObjective,
    QuantileObjective,
    SampleObjective,
)

WEIGHTS: PretrainedWeights = PretrainedWeights(repository="amazon/chronos-bolt-tiny", revision="main")
PRETRAINED: ChronosBoltParameters = replace(TINY, weights=WEIGHTS)
LEVELS: QuantileObjective = QuantileObjective((0.1, 0.5, 0.9))

type PublishedFile = tuple[dict[str, Tensor], list[PretrainedWeights]]


@pytest.fixture
def published_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PublishedFile:
    """Serve a weight file written under the published names in place of the Hugging Face Hub."""
    path: Path = tmp_path / "model.safetensors"
    tensors: dict[str, Tensor] = published_state_dict(TINY)
    SafetensorsFile.write(path, tensors)
    requests: list[PretrainedWeights] = []

    def download(weights: PretrainedWeights) -> Path:
        requests.append(weights)
        return path

    monkeypatch.setattr(HubClient, "download", staticmethod(download))
    return tensors, requests


def build(
    model: ForecastingModel = PRETRAINED,
    objective: ForecastObjective = LEVELS,
    training: TrainingSettings | None = None,
) -> Forecaster:
    return ForecasterSettings(
        model=model,
        window=ForecastWindow(input_length=40, horizon=7, stride=5),
        objective=objective,
        target_channels=(1, 0),
        scaling=NoScalingParameters(),
        is_instance_normalized=False,
        training=training or TrainingSettings(device=Device.CPU, max_epochs=1, batch_size=8),
    ).build()


def test_pretrained_model_forecasts_without_fit(published_file: PublishedFile, sine_series: np.ndarray) -> None:
    tensors, requests = published_file
    forecaster: Forecaster = build()
    forecaster.prepare_zero_shot(sine_series)
    assert requests == [WEIGHTS]
    assert forecaster.history.num_epochs == 0
    network = forecaster.network
    assert isinstance(network, ChronosBoltForecastNetwork)
    assert not network.training
    torch.testing.assert_close(network.model.token_embedding.weight, tensors["shared.weight"])
    prediction = forecaster.forecast(sine_series)
    assert prediction.quantiles.shape == (1, 7, 2, 3)
    # Without scaling and instance normalization the network sees the raw values of each target channel.
    raw: Tensor = torch.from_numpy(sine_series[-40:, [1, 0]].T.astype(np.float32))
    with torch.no_grad():
        expected: Tensor = network.forecast(raw).permute(2, 0, 1)
    # QuantileObjective sorts the quantiles, which the random weights of the test let cross.
    np.testing.assert_allclose(prediction.quantiles[0], expected.sort(dim=-1).values.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prediction.point, prediction.get_quantile(0.5))


def test_zero_shot_forecaster_loads_without_downloading(
    published_file: PublishedFile, sine_series: np.ndarray, tmp_path: Path
) -> None:
    _, requests = published_file
    forecaster: Forecaster = build()
    forecaster.prepare_zero_shot(sine_series)
    path: Path = tmp_path / "zero_shot.pt"
    forecaster.save(path)
    loaded: Forecaster = Forecaster.load(path)
    assert requests == [WEIGHTS]
    np.testing.assert_allclose(loaded.forecast(sine_series).quantiles, forecaster.forecast(sine_series).quantiles)


def test_point_objective_forecasts_the_median(published_file: PublishedFile, sine_series: np.ndarray) -> None:
    forecaster: Forecaster = build(objective=PointObjective())
    forecaster.prepare_zero_shot(sine_series)
    network = forecaster.network
    assert isinstance(network, ChronosBoltForecastNetwork)
    raw: Tensor = torch.from_numpy(sine_series[-40:, [1, 0]].T.astype(np.float32))
    with torch.no_grad():
        median: Tensor = network.forecast(raw)[:, 1].T
    np.testing.assert_allclose(forecaster.forecast(sine_series).point[0], median.numpy(), rtol=1e-5, atol=1e-5)


def test_fit_fine_tunes_from_the_published_weights(published_file: PublishedFile, sine_series: np.ndarray) -> None:
    tensors, requests = published_file
    training = TrainingSettings(device=Device.CPU, max_epochs=1, batch_size=8, learning_rate=1e-9)
    forecaster: Forecaster = build(training=training)
    history = forecaster.fit(sine_series[:200])
    assert history.num_epochs == 1
    assert requests == [WEIGHTS]
    network = forecaster.network
    assert isinstance(network, ChronosBoltForecastNetwork)
    torch.testing.assert_close(network.model.token_embedding.weight, tensors["shared.weight"], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("objective", [GaussianObjective(), QuantileObjective((0.25, 0.5, 0.75))])
def test_objective_must_match_the_model_outputs(
    objective: ForecastObjective,
    published_file: PublishedFile,
    sine_series: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="zero-shot"):
        build(objective=objective).prepare_zero_shot(sine_series)
    assert not published_file[1]


def test_models_without_published_weights_are_rejected(sine_series: np.ndarray) -> None:
    with pytest.raises(ValueError, match="published weights"):
        build(model=TINY).prepare_zero_shot(sine_series)
    with pytest.raises(TypeError, match="zero-shot"):
        build(model=MLPParameters(), objective=PointObjective()).prepare_zero_shot(sine_series)


def test_generative_models_forecast_samples_zero_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sine_series: np.ndarray
) -> None:
    architecture = ChronosParameters(
        model_size=16,
        key_value_size=4,
        num_heads=2,
        feedforward_size=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_tokens=64,
        context_length=12,
        prediction_length=4,
    )
    _ = torch.manual_seed(0)
    path: Path = tmp_path / "model.safetensors"
    # The network's own names pass through the published-name mapping unchanged.
    SafetensorsFile.write(path, architecture.build_forecast_network(build_shape()).state_dict())

    def download(weights: PretrainedWeights) -> Path:
        return path

    monkeypatch.setattr(HubClient, "download", staticmethod(download))
    model = replace(architecture, weights=WEIGHTS)
    with pytest.raises(ValueError, match="SampleObjective"):
        build(model=model).prepare_zero_shot(sine_series)
    forecaster: Forecaster = build(model=model, objective=SampleObjective(num_samples=5))
    forecaster.prepare_zero_shot(sine_series)
    prediction = forecaster.forecast(sine_series, num_samples=5)
    assert prediction.point.shape == (1, 7, 2)
    assert prediction.samples is not None
    assert prediction.samples.shape == (1, 5, 7, 2)


def build_shape() -> ForecastShape:
    return ForecastShape(
        input_length=40, input_channels=3, horizon=7, target_channels=(1, 0), num_parameters=5, level_parameters=(0,)
    )
