import math

import pytest
import torch
import torch.nn.functional as functional
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, EncoderHead, ForecastShape, SequenceShape, SummaryShape, TemporalPooling
from time_series_dl.models.architectures.transformer.patch_tst import TokenNormalization
from time_series_dl.models.architectures.transformer.tst import TSTEncoderLayer, TSTParameters, TSTSequenceEncoder

MODELS: list[TSTParameters] = [
    TSTParameters(model_size=8, num_heads=2, num_layers=2, feedforward_size=16),
    TSTParameters(
        model_size=6,
        num_heads=3,
        num_layers=1,
        feedforward_size=8,
        activation=Activation.RELU,
        normalization=TokenNormalization.LAYER,
        head=EncoderHead(pooling=TemporalPooling.MEAN),
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: TSTParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: TSTParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: TSTParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_encoder_matches_reference_computation() -> None:
    _ = torch.manual_seed(0)
    encoder: TSTSequenceEncoder = TSTSequenceEncoder(MODELS[0], 7, 3).eval()
    values: Tensor = torch.randn(2, 7, 3)
    with torch.no_grad():
        tokens: Tensor = encoder.input_projection(values) * math.sqrt(8) + encoder.position
        expected: Tensor = functional.gelu(encoder.layers(tokens))
        assert torch.allclose(encoder(values), expected, atol=1e-6)
    assert encoder.output_length == 7
    assert encoder.output_size == 8


@pytest.mark.parametrize("normalization", list(TokenNormalization))
def test_layer_normalizes_over_the_configured_dimensions(normalization: TokenNormalization) -> None:
    _ = torch.manual_seed(0)
    parameters: TSTParameters = TSTParameters(
        model_size=8, num_heads=2, feedforward_size=16, dropout=0.0, normalization=normalization
    )
    layer: TSTEncoderLayer = TSTEncoderLayer(parameters).train()
    outputs: Tensor = layer(torch.randn(4, 10, 8) * 3.0 + 1.0)
    match normalization:
        case TokenNormalization.BATCH:
            assert torch.allclose(outputs.mean(dim=(0, 1)), torch.zeros(8), atol=1e-5)
            assert torch.allclose(outputs.var(dim=(0, 1), unbiased=False), torch.ones(8), atol=1e-3)
        case TokenNormalization.LAYER:
            assert torch.allclose(outputs.mean(dim=-1), torch.zeros(4, 10), atol=1e-5)


def test_default_head_flattens_every_step() -> None:
    assert TSTParameters().head == EncoderHead(pooling=TemporalPooling.FLATTEN, dropout=0.1)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        TSTParameters(num_layers=0)
    with pytest.raises(ValueError, match="num_heads"):
        TSTParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        TSTParameters(dropout=1.0)
