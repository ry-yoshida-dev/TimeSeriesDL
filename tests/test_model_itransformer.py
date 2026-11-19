import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.itransformer import ITransformerEncoder, ITransformerParameters

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
    ForecastShape(input_length=15, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

MODELS: list[ITransformerParameters] = [
    ITransformerParameters(model_size=8, num_heads=2, num_layers=2, feedforward_size=16),
    ITransformerParameters(model_size=8, num_heads=1, num_layers=1, feedforward_size=8, activation=Activation.RELU),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ITransformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: ITransformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: ITransformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("channels", [1, 5])
def test_one_token_per_channel(channels: int) -> None:
    encoder: ITransformerEncoder = ITransformerEncoder(MODELS[0], 13)
    assert encoder(torch.randn(2, 13, channels)).shape == (2, channels, 8)


def test_forecast_follows_channel_permutation() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=12, input_channels=3, horizon=2, target_channels=(0, 1, 2))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 12, 3)
    permutation: list[int] = [2, 0, 1]
    with torch.no_grad():
        original: Tensor = network(past)
        permuted: Tensor = network(past[..., permutation])
    assert torch.allclose(original[:, :, permutation], permuted, atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        ITransformerParameters(num_layers=0)
    with pytest.raises(ValueError, match="num_heads"):
        ITransformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        ITransformerParameters(dropout=-0.1)
