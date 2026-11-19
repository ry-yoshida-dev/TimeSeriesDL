from pathlib import Path

import numpy as np
import pytest
import torch
from data_scaler import NoScalingParameters
from torch import Tensor, nn

from time_series_dl import Device, TrainingSettings
from time_series_dl.data import SeriesCovariates
from time_series_dl.models import ForecastShape, ModelArchitecture, NLinearParameters
from time_series_dl.models.architectures.linear.cycle_net import (
    CycleNetBackbone,
    CycleNetCovariateForecastNetwork,
    CycleNetParameters,
    CycleNetRecurrentCycle,
)
from time_series_dl.models.interfaces import CovariateForecastShape, CovariateTensors
from time_series_dl.tasks.forecasting import Forecaster, ForecasterSettings, ForecastWindow

BATCH_SIZE: int = 3

COVARIATE_SHAPES: list[CovariateForecastShape] = [
    CovariateForecastShape(
        forecast=ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
        known_channels=1,
        static_channels=2,
    ),
    CovariateForecastShape(
        forecast=ForecastShape(
            input_length=17,
            input_channels=4,
            horizon=5,
            target_channels=(2, 0),
            num_parameters=3,
            location_index=1,
            level_parameters=(0, 1, 2),
        ),
        known_channels=3,
        static_cardinalities=(4,),
    ),
    CovariateForecastShape(
        forecast=ForecastShape(input_length=3, input_channels=1, horizon=9, target_channels=(0,), num_parameters=2),
        known_channels=2,
    ),
]

MODELS: list[CycleNetParameters] = [
    CycleNetParameters(cycle_length=7, backbone=CycleNetBackbone.LINEAR),
    CycleNetParameters(cycle_length=5, backbone=CycleNetBackbone.MLP, hidden_size=8, phase_channel=1),
]


def make_covariates(
    shape: CovariateForecastShape, cycle_length: int, phase_channel: int, starts: Tensor
) -> CovariateTensors:
    """Covariates whose phase channel counts up from each window's start phase; the other channels are noise."""
    length: int = shape.forecast.input_length + shape.forecast.horizon
    known: Tensor = torch.randn(starts.shape[0], length, shape.known_channels)
    known[:, :, phase_channel] = ((starts.unsqueeze(1) + torch.arange(length)) % cycle_length).float()
    return CovariateTensors(
        known=known,
        static=torch.randn(starts.shape[0], shape.static_channels),
        static_categories=torch.zeros(starts.shape[0], shape.static_categories, dtype=torch.int64),
    )


@pytest.mark.parametrize("shape", COVARIATE_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_covariate_forecast_network(model: CycleNetParameters, shape: CovariateForecastShape) -> None:
    if model.phase_channel >= shape.known_channels:
        with pytest.raises(ValueError, match="known covariate channel"):
            _ = model.build_covariate_forecast_network(shape)
        return
    _ = torch.manual_seed(0)
    network = model.build_covariate_forecast_network(shape)
    forecast: ForecastShape = shape.forecast
    past: Tensor = torch.randn(BATCH_SIZE, forecast.input_length, forecast.input_channels)
    future: Tensor = torch.randn(BATCH_SIZE, forecast.horizon, forecast.output_channels)
    covariates = make_covariates(shape, model.cycle_length, model.phase_channel, torch.tensor([0, 3, 6]))
    outputs: Tensor = network.train()(past, covariates, future)
    assert outputs.shape == (BATCH_SIZE, forecast.horizon, forecast.output_channels, forecast.num_parameters)
    outputs.square().mean().backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    with torch.no_grad():
        inferred: Tensor = network.eval()(past, covariates)
    assert bool(torch.isfinite(inferred).all())


def test_recurrent_cycle_matches_the_official_index_arithmetic() -> None:
    # The official code gathers the cycle at (start + step) % W for the window and at (start + L + step) % W for the
    # horizon; a phase covariate counting up from the window's start phase must give the same segments.
    cycle_length, input_length, horizon, channels = 5, 7, 4, 2
    shape = CovariateForecastShape(
        forecast=ForecastShape(
            input_length=input_length,
            input_channels=channels,
            horizon=horizon,
            target_channels=(0, 1),
            num_parameters=2,
            level_parameters=(0,),
        ),
        known_channels=1,
    )
    network = CycleNetParameters(
        cycle_length=cycle_length, backbone=CycleNetBackbone.LINEAR
    ).build_covariate_forecast_network(shape)
    assert isinstance(network, CycleNetCovariateForecastNetwork)
    _ = torch.manual_seed(1)
    with torch.no_grad():
        _ = network.recurrent_cycle.cycle.copy_(torch.randn(cycle_length, channels))
    starts: Tensor = torch.tensor([0, 3, 4])
    covariates = make_covariates(shape, cycle_length, 0, starts)
    past: Tensor = torch.randn(3, input_length, channels)
    cycle: Tensor = network.recurrent_cycle.cycle.detach()
    past_cycle: Tensor = cycle[(starts.view(-1, 1) + torch.arange(input_length).view(1, -1)) % cycle_length]
    future_cycle: Tensor = cycle[
        (((starts + input_length) % cycle_length).view(-1, 1) + torch.arange(horizon).view(1, -1)) % cycle_length
    ]
    backbone = network.backbone
    assert isinstance(backbone, nn.Linear)
    mapped: Tensor = backbone((past - past_cycle).transpose(1, 2)).reshape(3, channels, horizon, 2).transpose(1, 2)
    expected: Tensor = mapped.clone()
    expected[..., 0] += future_cycle
    with torch.no_grad():
        np.testing.assert_allclose(network(past, covariates).numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
    lookup = CycleNetRecurrentCycle(cycle_length, channels)
    with torch.no_grad():
        _ = lookup.cycle.copy_(cycle)
    phase: Tensor = torch.tensor([[4, 0, 1]])
    assert torch.equal(lookup(phase), cycle[[4, 0, 1]].unsqueeze(0))


def test_phase_must_be_an_unscaled_integer_in_range() -> None:
    shape = CovariateForecastShape(
        forecast=ForecastShape(input_length=4, input_channels=1, horizon=2, target_channels=(0,)), known_channels=1
    )
    network = CycleNetParameters(cycle_length=3).build_covariate_forecast_network(shape)
    past: Tensor = torch.randn(1, 4, 1)
    for phase in (torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0]), torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])):
        covariates = CovariateTensors(
            known=phase.view(1, 6, 1), static=torch.zeros(1, 0), static_categories=torch.zeros(1, 0, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="NoScalingParameters"):
            _ = network(past, covariates)


def test_parameters_reject_invalid_values_and_plain_forecasts() -> None:
    with pytest.raises(ValueError, match="cycle_length"):
        CycleNetParameters(cycle_length=0)
    with pytest.raises(ValueError, match="hidden_size"):
        CycleNetParameters(hidden_size=0)
    with pytest.raises(ValueError, match="phase_channel"):
        CycleNetParameters(phase_channel=-1)
    with pytest.raises(ValueError, match="phase"):
        _ = CycleNetParameters().build_forecast_network(
            ForecastShape(input_length=4, input_channels=1, horizon=2, target_channels=(0,))
        )


WINDOW: ForecastWindow = ForecastWindow(input_length=24, horizon=12)
WEEK: int = 168


def make_weekly_series(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Hourly values of a random weekly profile with small noise, and the hour of the week of each step."""
    rng = np.random.default_rng(0)
    profile = rng.normal(size=WEEK)
    steps = np.arange(length)
    values = profile[steps % WEEK] + rng.normal(scale=0.1, size=length)
    return values[:, np.newaxis], (steps % WEEK).astype(np.float64)[:, np.newaxis]


def build_forecaster(backbone: CycleNetBackbone) -> Forecaster:
    training = TrainingSettings(
        max_epochs=10, batch_size=64, learning_rate=1e-2, device=Device.CPU, early_stopping_patience=None, random_seed=0
    )
    return ForecasterSettings(
        model=CycleNetParameters(cycle_length=WEEK, backbone=backbone, hidden_size=64),
        window=WINDOW,
        training=training,
        covariate_scaling=NoScalingParameters(),
    ).build()


@pytest.mark.parametrize("backbone", list(CycleNetBackbone))
def test_weekly_cycle_is_learned_from_the_phase(backbone: CycleNetBackbone) -> None:
    # A window of one day cannot tell which day of the week follows; the phase covariate can.
    series, phase = make_weekly_series(WEEK * 12)
    split: int = WEEK * 9
    forecaster = build_forecaster(backbone)
    _ = forecaster.fit(series[:split], covariates=SeriesCovariates(known=phase[:split]))
    assert isinstance(forecaster.network, CycleNetCovariateForecastNetwork)
    error: float = forecaster.evaluate(series[split:], SeriesCovariates(known=phase[split:])).mae
    blind = ForecasterSettings(model=NLinearParameters(), window=WINDOW, training=forecaster.settings.training).build()
    _ = blind.fit(series[:split])
    assert blind.evaluate(series[split:]).mae > 0.6
    assert error < 0.3


def test_forecaster_needs_the_phase_covariate(fast_training: TrainingSettings) -> None:
    series, _ = make_weekly_series(200)
    forecaster = ForecasterSettings(model=CycleNetParameters(), window=WINDOW, training=fast_training).build()
    with pytest.raises(ValueError, match="phase"):
        _ = forecaster.fit(series)


@pytest.mark.skipif(
    "cycle_net" not in {architecture.value for architecture in ModelArchitecture},
    reason="CycleNet is not registered in ModelArchitecture yet, so its settings cannot be decoded.",
)
def test_forecaster_round_trip(tmp_path: Path, fast_training: TrainingSettings) -> None:
    series, phase = make_weekly_series(WEEK * 3)
    forecaster = ForecasterSettings(
        model=CycleNetParameters(cycle_length=WEEK, hidden_size=16),
        window=WINDOW,
        training=fast_training,
        covariate_scaling=NoScalingParameters(),
    ).build()
    _ = forecaster.fit(series, covariates=SeriesCovariates(known=phase))
    path = tmp_path / "cycle_net.pt"
    forecaster.save(path)
    loaded = Forecaster.load(path)
    assert loaded.settings == forecaster.settings
    covariates = SeriesCovariates(known=np.arange(series.shape[0] + WINDOW.horizon, dtype=np.float64)[:, None] % WEEK)
    np.testing.assert_allclose(
        loaded.forecast(series, covariates).point, forecaster.forecast(series, covariates).point, rtol=1e-6
    )
