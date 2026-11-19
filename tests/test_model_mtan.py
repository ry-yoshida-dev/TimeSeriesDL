import math

import pytest
import torch
from irregular_network_checks import (
    CHANNELS,
    check_irregular_interpolation_network,
    check_irregular_summary_network,
    make_sequences,
    read_preset_parameters,
)
from torch import Tensor

from time_series_dl.models.architectures.continuous_time.mtan import (
    LearnedTimeEmbedding,
    MTANParameters,
    MultiTimeAttention,
)
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors
from time_series_dl.models.settings.parameters_reader import ParametersReader

MODELS: list[MTANParameters] = [
    MTANParameters(hidden_size=8, time_embedding_size=8, num_reference_points=6, latent_size=3, projection_size=5),
    MTANParameters(
        hidden_size=6,
        time_embedding_size=12,
        num_heads=3,
        num_reference_points=4,
        latent_size=2,
        projection_size=4,
        head=IrregularEncoderHead(hidden_sizes=(5,)),
    ),
]


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MTANParameters) -> None:
    _ = check_irregular_summary_network(model)


@pytest.mark.parametrize("model", MODELS)
def test_interpolation_network(model: MTANParameters) -> None:
    _ = check_irregular_interpolation_network(model)


def test_attention_matches_an_explicit_computation() -> None:
    _ = torch.manual_seed(0)
    num_heads: int = 2
    attention = MultiTimeAttention(value_size=3, output_size=4, embedding_size=6, num_heads=num_heads).double()
    queries: Tensor = torch.randn(1, 5, 6, dtype=torch.float64)
    keys: Tensor = torch.randn(2, 7, 6, dtype=torch.float64)
    values: Tensor = torch.randn(2, 7, 3, dtype=torch.float64)
    mask: Tensor = (torch.rand(2, 7, 3) < 0.5).double()
    mask[1, :, 2] = 0.0
    head_size: int = 3
    with torch.no_grad():
        outputs: Tensor = attention(queries, keys, values, mask)
        projected_queries: Tensor = attention.query_projection(queries)
        projected_keys: Tensor = attention.key_projection(keys)
    expected: Tensor = torch.zeros(2, 5, num_heads * 3, dtype=torch.float64)
    for sample in range(2):
        for query in range(5):
            for head in range(num_heads):
                part: slice = slice(head * head_size, (head + 1) * head_size)
                for channel in range(3):
                    observed: list[int] = [key for key in range(7) if mask[sample, key, channel] > 0]
                    if not observed:
                        continue
                    scores: list[float] = [
                        float(projected_queries[0, query, part] @ projected_keys[sample, key, part])
                        / math.sqrt(head_size)
                        for key in observed
                    ]
                    weights: Tensor = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0)
                    expected[sample, query, head * 3 + channel] = (weights * values[sample, observed, channel]).sum()
    with torch.no_grad():
        assert torch.allclose(outputs, attention.output_projection(expected), atol=1e-10)


def test_attention_without_mask_is_a_softmax_over_every_key() -> None:
    _ = torch.manual_seed(1)
    attention = MultiTimeAttention(value_size=2, output_size=2, embedding_size=4, num_heads=1)
    keys: Tensor = torch.randn(3, 5, 4)
    values: Tensor = torch.randn(3, 5, 2)
    queries: Tensor = torch.randn(3, 2, 4)
    assert torch.allclose(attention(queries, keys, values), attention(queries, keys, values, torch.ones_like(values)))


def test_encoder_ignores_the_order_of_the_observations() -> None:
    _ = torch.manual_seed(0)
    network = MODELS[0].build_encoder(CHANNELS).eval()
    sequences: IrregularTensors = make_sequences(lengths=(6, 6))
    order: Tensor = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled = IrregularTensors(
        times=sequences.times[:, order],
        values=sequences.values[:, order],
        observed_mask=sequences.observed_mask[:, order],
        lengths=sequences.lengths,
    )
    with torch.no_grad():
        assert torch.allclose(network(sequences), network(shuffled), atol=1e-6)


def test_time_embedding_is_linear_then_periodic() -> None:
    embedding = LearnedTimeEmbedding(5)
    times: Tensor = torch.tensor([[0.0, 0.5, 1.0]])
    embedded: Tensor = embedding(times)
    assert embedded.shape == (1, 3, 5)
    assert torch.allclose(embedded[..., 0], embedding.linear.weight[0, 0] * times + embedding.linear.bias[0])
    assert bool((embedded[..., 1:].abs() <= 1.0).all())
    with pytest.raises(ValueError, match="at least 2"):
        _ = LearnedTimeEmbedding(1)


def test_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _ = MTANParameters(time_embedding_size=10, num_heads=3)
    with pytest.raises(ValueError, match="observation_std"):
        _ = MTANParameters(observation_std=0.0)
    with pytest.raises(ValueError, match="num_reference_points"):
        _ = MTANParameters(num_reference_points=0)


def test_preset_holds_the_defaults() -> None:
    assert ParametersReader.read(MTANParameters, read_preset_parameters("mtan")) == MTANParameters()
