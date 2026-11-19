import math

import pytest
import torch
import torch.nn.functional as functional
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.conv_tran import (
    AbsolutePositionEncoding,
    ConvTranAttention,
    ConvTranConvolutionalEmbedding,
    ConvTranParameters,
    ConvTranPositionEncoding,
    RelativePositionEncoding,
)

MODELS: list[ConvTranParameters] = [
    ConvTranParameters(model_size=8, num_heads=2, feedforward_size=16),
    ConvTranParameters(
        model_size=6,
        num_heads=3,
        feedforward_size=8,
        kernel_size=3,
        dropout=0.0,
        absolute_position_encoding=AbsolutePositionEncoding.LEARNABLE,
        relative_position_encoding=RelativePositionEncoding.NONE,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ConvTranParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: ConvTranParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("encoding", list(AbsolutePositionEncoding))
def test_summary_network(encoding: AbsolutePositionEncoding) -> None:
    model: ConvTranParameters = ConvTranParameters(
        model_size=8, num_heads=2, feedforward_size=16, absolute_position_encoding=encoding
    )
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("model_size", [6, 7])
def test_tape_follows_its_formula(model_size: int) -> None:
    length: int = 5
    encoding: ConvTranPositionEncoding = ConvTranPositionEncoding(
        AbsolutePositionEncoding.TAPE, model_size, length, dropout=0.0
    )
    expected: Tensor = torch.zeros(length, model_size)
    for position in range(length):
        for feature in range(model_size):
            frequency: float = math.exp(-(feature - feature % 2) * math.log(10000.0) / model_size)
            angle: float = position * frequency * model_size / length
            expected[position, feature] = math.sin(angle) if feature % 2 == 0 else math.cos(angle)
    assert torch.allclose(encoding.encoding, expected, atol=1e-5)
    tokens: Tensor = torch.randn(2, length, model_size)
    assert torch.allclose(encoding(tokens), tokens + expected, atol=1e-5)


def test_no_absolute_encoding_is_identity() -> None:
    encoding: ConvTranPositionEncoding = ConvTranPositionEncoding(AbsolutePositionEncoding.NONE, 4, 3, dropout=0.5)
    tokens: Tensor = torch.randn(2, 3, 4)
    assert torch.equal(encoding.train()(tokens), tokens)


def test_erpe_attention_matches_reference_computation() -> None:
    _ = torch.manual_seed(0)
    length: int = 4
    model_size: int = 6
    num_heads: int = 2
    attention: ConvTranAttention = ConvTranAttention(RelativePositionEncoding.ERPE, model_size, num_heads, length, 0.0)
    table = attention.relative_bias_table
    assert table is not None
    with torch.no_grad():
        _ = table.normal_()
    tokens: Tensor = torch.randn(2, length, model_size)
    head_size: int = model_size // num_heads
    queries: Tensor = tokens @ attention.query_projection.weight.T
    keys: Tensor = tokens @ attention.key_projection.weight.T
    values: Tensor = tokens @ attention.value_projection.weight.T
    expected: Tensor = torch.zeros(2, length, model_size)
    for head in range(num_heads):
        features: slice = slice(head * head_size, (head + 1) * head_size)
        scores: Tensor = queries[..., features] @ keys[..., features].transpose(1, 2) / math.sqrt(model_size)
        weights: Tensor = torch.softmax(scores, dim=-1)
        for query in range(length):
            for key in range(length):
                weights[:, query, key] += table[query - key + length - 1, head]
        expected[..., features] = weights @ values[..., features]
    expected = functional.layer_norm(expected, (model_size,), attention.output_norm.weight, attention.output_norm.bias)
    with torch.no_grad():
        assert torch.allclose(attention(tokens), expected, atol=1e-5)


def test_attention_without_erpe_has_no_bias_table() -> None:
    attention: ConvTranAttention = ConvTranAttention(RelativePositionEncoding.NONE, 4, 2, 3, 0.0)
    assert attention.relative_bias_table is None
    assert attention.compute_relative_bias() is None


@pytest.mark.parametrize("kernel_size", [3, 8])
def test_embedding_keeps_the_length(kernel_size: int) -> None:
    embedding: ConvTranConvolutionalEmbedding = ConvTranConvolutionalEmbedding(3, 8, kernel_size)
    assert embedding(torch.randn(2, 11, 3)).shape == (2, 11, 8)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ConvTranParameters(kernel_size=0)
    with pytest.raises(ValueError, match="num_heads"):
        ConvTranParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        ConvTranParameters(dropout=-0.1)
