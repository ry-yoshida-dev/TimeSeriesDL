import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SMambaParameters, SummaryShape
from time_series_dl.models.architectures.state_space.s_mamba import SMambaEncoder

# The Mamba blocks scan over the variates, so a single variate leaves their state matrices without a gradient; the
# gradient checks use several channels.
FORECAST_SHAPES: list[ForecastShape] = [
    ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
    ForecastShape(
        input_length=17,
        input_channels=4,
        horizon=5,
        target_channels=(2, 0),
        num_parameters=3,
        location_index=1,
        level_parameters=(0, 1, 2),
    ),
    ForecastShape(input_length=15, input_channels=2, horizon=1, target_channels=(1,), num_parameters=2),
]

MODELS: list[SMambaParameters] = [
    SMambaParameters(model_size=8, state_size=4, num_layers=2, feedforward_size=16),
    SMambaParameters(model_size=4, state_size=2, num_layers=1, feedforward_size=4, activation=Activation.RELU),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: SMambaParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: SMambaParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: SMambaParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("channels", [1, 5])
def test_one_token_per_channel(channels: int) -> None:
    encoder: SMambaEncoder = SMambaEncoder(MODELS[0], 13)
    assert encoder(torch.randn(2, 13, channels)).shape == (2, channels, 8)


def test_univariate_forecast() -> None:
    shape: ForecastShape = ForecastShape(input_length=10, input_channels=1, horizon=3, target_channels=(0,))
    network = MODELS[0].build_forecast_network(shape).eval()
    with torch.no_grad():
        outputs: Tensor = network(torch.randn(2, 10, 1))
    assert outputs.shape == (2, 3, 1, 1)
    assert bool(torch.isfinite(outputs).all())


def test_every_variate_sees_the_others() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=2, target_channels=(0, 1, 2))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, :, 1] += 1.0
    with torch.no_grad():
        original: Tensor = network(past)
        outputs: Tensor = network(changed)
    assert not torch.allclose(original[:, :, 0], outputs[:, :, 0])
    assert not torch.allclose(original[:, :, 2], outputs[:, :, 2])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        SMambaParameters(model_size=0)
    with pytest.raises(ValueError, match="feedforward_size"):
        SMambaParameters(feedforward_size=0)
    with pytest.raises(ValueError, match="dropout"):
        SMambaParameters(dropout=1.0)
