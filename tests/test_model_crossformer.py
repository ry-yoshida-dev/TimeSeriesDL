import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.crossformer import (
    CrossformerParameters,
    CrossformerSegmentEmbedding,
    CrossformerSegmentMerging,
    CrossformerTwoStageAttentionLayer,
)

MODELS: list[CrossformerParameters] = [
    CrossformerParameters(
        segment_length=4, num_routers=3, model_size=8, num_heads=2, num_layers=2, feedforward_size=16, dropout=0.1
    ),
    CrossformerParameters(
        segment_length=3,
        merge_size=3,
        num_routers=2,
        model_size=6,
        num_heads=3,
        num_layers=3,
        feedforward_size=8,
        activation=Activation.RELU,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: CrossformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: CrossformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: CrossformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_scale_segments_merge_with_rounding_up() -> None:
    assert MODELS[0].count_scale_segments(19) == (5, 5, 3)
    assert MODELS[1].count_scale_segments(19) == (7, 7, 3, 1)


def test_segment_embedding_pads_with_the_first_value() -> None:
    _ = torch.manual_seed(0)
    embedding = CrossformerSegmentEmbedding(input_length=6, input_channels=2, segment_length=4, model_size=5)
    values: Tensor = torch.randn(3, 6, 2)
    padded: Tensor = torch.cat([values[:, :1], values[:, :1], values], dim=1)
    segments: Tensor = padded.transpose(1, 2).reshape(3, 2, 2, 4)
    expected: Tensor = embedding.norm(embedding.projection(segments) + embedding.position)
    assert embedding.num_segments == 2
    assert torch.allclose(embedding(values), expected)


def test_segment_merging_concatenates_neighbours_and_repeats_the_last() -> None:
    merging = CrossformerSegmentMerging(num_segments=5, merge_size=2, model_size=3)
    tokens: Tensor = torch.randn(2, 4, 5, 3)
    padded: Tensor = torch.cat([tokens, tokens[:, :, -1:]], dim=2)
    concatenated: Tensor = torch.cat([padded[:, :, 0::2], padded[:, :, 1::2]], dim=-1)
    expected: Tensor = merging.projection(merging.norm(concatenated))
    assert torch.allclose(merging(tokens), expected)


def test_two_stage_attention_is_channel_equivariant() -> None:
    _ = torch.manual_seed(0)
    layer = CrossformerTwoStageAttentionLayer(
        num_segments=3,
        num_routers=2,
        model_size=8,
        num_heads=2,
        feedforward_size=16,
        dropout=0.0,
        activation=Activation.GELU,
    ).eval()
    tokens: Tensor = torch.randn(2, 5, 3, 8)
    permutation: list[int] = [3, 0, 4, 1, 2]
    with torch.no_grad():
        original: Tensor = layer(tokens)
        permuted: Tensor = layer(tokens[:, permutation])
    assert original.shape == tokens.shape
    assert torch.allclose(original[:, permutation], permuted, atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="segment_length"):
        CrossformerParameters(segment_length=0)
    with pytest.raises(ValueError, match="merge_size"):
        CrossformerParameters(merge_size=1)
    with pytest.raises(ValueError, match="num_heads"):
        CrossformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        CrossformerParameters(dropout=1.0)
