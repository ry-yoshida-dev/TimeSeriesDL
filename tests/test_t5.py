import math

import pytest
import torch
from t5_published_names import published_stack_names
from torch import Tensor

from time_series_dl.models.components.pretrained import ParameterNameMapping
from time_series_dl.models.components.t5 import (
    T5Configuration,
    T5Decoder,
    T5Encoder,
    T5FeedForwardKind,
    T5LayerNorm,
    T5OfficialNames,
    T5RelativePositionBias,
)


def configuration(kind: T5FeedForwardKind = T5FeedForwardKind.RELU) -> T5Configuration:
    return T5Configuration(
        model_size=8,
        key_value_size=3,
        num_heads=2,
        feedforward_size=12,
        num_layers=2,
        feed_forward_kind=kind,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
        dropout=0.0,
    )


def reference_bucket(relative_position: int, is_bidirectional: bool, num_buckets: int, max_distance: int) -> int:
    """The bucketing of the T5 paper's reference implementation, one position at a time."""
    bucket: int = 0
    if is_bidirectional:
        num_buckets //= 2
        bucket += num_buckets if relative_position > 0 else 0
        distance: int = abs(relative_position)
    else:
        distance = max(-relative_position, 0)
    max_exact: int = num_buckets // 2
    if distance < max_exact:
        return bucket + distance
    fraction: float = math.log(distance / max_exact) / math.log(max_distance / max_exact)
    large: int = max_exact + int(fraction * (num_buckets - max_exact))
    return bucket + min(large, num_buckets - 1)


@pytest.mark.parametrize("is_bidirectional", [True, False])
def test_relative_position_buckets_follow_the_reference(is_bidirectional: bool) -> None:
    bias = T5RelativePositionBias(num_heads=2, num_buckets=32, max_distance=128, is_bidirectional=is_bidirectional)
    positions: list[int] = [-1000, -200, -40, -20, -9, -8, -3, -1, 0, 1, 3, 8, 9, 20, 40, 200, 1000]
    buckets: Tensor = bias.compute_buckets(torch.tensor(positions))
    expected: list[int] = [reference_bucket(position, is_bidirectional, 32, 128) for position in positions]
    assert buckets.tolist() == expected


def test_relative_position_bias_indexes_the_embedding_by_bucket() -> None:
    bias = T5RelativePositionBias(num_heads=2, num_buckets=8, max_distance=16, is_bidirectional=True)
    values: Tensor = bias(3, 5)
    assert values.shape == (1, 2, 3, 5)
    bucket: int = reference_bucket(4 - 1, True, 8, 16)
    torch.testing.assert_close(values[0, :, 1, 4], bias.embedding.weight[bucket])


def test_layer_norm_scales_by_root_mean_square_without_centering() -> None:
    norm = T5LayerNorm(4, epsilon=1e-6)
    with torch.no_grad():
        _ = norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    values: Tensor = torch.tensor([[1.0, -2.0, 3.0, 5.0]])
    expected: Tensor = values / math.sqrt((1 + 4 + 9 + 25) / 4 + 1e-6) * norm.weight
    torch.testing.assert_close(norm(values), expected)


@pytest.mark.parametrize("kind", list(T5FeedForwardKind))
def test_encoder_ignores_masked_positions(kind: T5FeedForwardKind) -> None:
    _ = torch.manual_seed(0)
    encoder = T5Encoder(configuration(kind)).eval()
    embeddings: Tensor = torch.randn(2, 6, 8)
    is_observed: Tensor = torch.tensor([[False, False, True, True, True, True], [True] * 6])
    changed: Tensor = embeddings.clone()
    changed[0, :2] = torch.randn(2, 8) * 10
    with torch.no_grad():
        outputs: Tensor = encoder(embeddings, is_observed)
        changed_outputs: Tensor = encoder(changed, is_observed)
    torch.testing.assert_close(outputs[0, 2:], changed_outputs[0, 2:])
    torch.testing.assert_close(outputs[1], changed_outputs[1])


def test_decoder_is_causal() -> None:
    _ = torch.manual_seed(0)
    decoder = T5Decoder(configuration(T5FeedForwardKind.GATED_GELU)).eval()
    embeddings: Tensor = torch.randn(2, 5, 8)
    encoder_states: Tensor = torch.randn(2, 4, 8)
    is_observed: Tensor = torch.ones(2, 4, dtype=torch.bool)
    changed: Tensor = embeddings.clone()
    changed[:, 3:] = torch.randn(2, 2, 8)
    with torch.no_grad():
        outputs: Tensor = decoder(embeddings, encoder_states, is_observed)
        changed_outputs: Tensor = decoder(changed, encoder_states, is_observed)
    torch.testing.assert_close(outputs[:, :3], changed_outputs[:, :3])
    assert not torch.allclose(outputs[:, 3:], changed_outputs[:, 3:])


@pytest.mark.parametrize("kind", list(T5FeedForwardKind))
def test_official_names_cover_both_stacks_exactly(kind: T5FeedForwardKind) -> None:
    encoder = T5Encoder(configuration(kind))
    decoder = T5Decoder(configuration(kind))
    encoder_mapping = ParameterNameMapping(rules=T5OfficialNames.encoder_rules("encoder.", "stack."))
    decoder_mapping = ParameterNameMapping(rules=T5OfficialNames.decoder_rules("decoder.", "stack."))
    encoder_names: list[str] = [
        encoder_mapping.rename(name) for name in published_stack_names("encoder.", 2, is_decoder=False, kind=kind)
    ]
    decoder_names: list[str] = [
        decoder_mapping.rename(name) for name in published_stack_names("decoder.", 2, is_decoder=True, kind=kind)
    ]
    assert sorted(encoder_names) == sorted(f"stack.{name}" for name in encoder.state_dict())
    assert sorted(decoder_names) == sorted(f"stack.{name}" for name in decoder.state_dict())
    assert encoder_mapping.rename("encoder.block.1.layer.0.SelfAttention.q.weight") == (
        "stack.blocks.1.self_attention.query.weight"
    )
    assert decoder_mapping.rename("decoder.block.0.layer.1.EncDecAttention.o.weight") == (
        "stack.blocks.0.cross_attention.output.weight"
    )


def test_configuration_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="model_size"):
        _ = T5Configuration(model_size=0, key_value_size=2, num_heads=1, feedforward_size=4, num_layers=1)
    with pytest.raises(ValueError, match="dropout"):
        _ = T5Configuration(model_size=4, key_value_size=2, num_heads=1, feedforward_size=4, num_layers=1, dropout=1.0)
    with pytest.raises(ValueError, match="buckets"):
        _ = T5Configuration(
            model_size=4,
            key_value_size=2,
            num_heads=1,
            feedforward_size=4,
            num_layers=1,
            relative_attention_num_buckets=2,
        )
