import pytest
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import nn

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.vanilla import TransformerParameters

MODELS: list[TransformerParameters] = [
    TransformerParameters(model_size=8, num_heads=2, num_layers=2, feedforward_size=16),
    TransformerParameters(
        model_size=6, num_heads=3, num_layers=1, feedforward_size=8, activation=Activation.RELU, is_pre_norm=True
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TransformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TransformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TransformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("is_pre_norm", [False, True])
def test_normalization_order(is_pre_norm: bool) -> None:
    parameters = TransformerParameters(model_size=8, num_heads=2, feedforward_size=16, is_pre_norm=is_pre_norm)
    encoder = parameters.build_encoder(12, 2)
    layer = encoder.layers.layers[0]
    assert isinstance(layer, nn.TransformerEncoderLayer)
    assert layer.norm_first == is_pre_norm
    assert isinstance(encoder.layers.norm, nn.LayerNorm) == is_pre_norm
    assert (encoder.output_length, encoder.output_size) == (12, 8)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_heads must divide"):
        TransformerParameters(model_size=10, num_heads=3)
    with pytest.raises(ValueError, match="num_layers"):
        TransformerParameters(num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        TransformerParameters(dropout=1.0)
