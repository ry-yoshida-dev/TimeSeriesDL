import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.non_stationary_transformer import (
    DeStationaryAttention,
    DeStationaryFactorLearner,
    NonStationaryFactorProjector,
    NonStationaryTransformerParameters,
    Stationarization,
)

MODELS: list[NonStationaryTransformerParameters] = [
    NonStationaryTransformerParameters(
        model_size=8, num_heads=2, feedforward_size=16, projector_hidden_sizes=(8, 8), label_length=2
    ),
    NonStationaryTransformerParameters(
        model_size=6,
        num_heads=3,
        num_encoder_layers=1,
        num_decoder_layers=2,
        feedforward_size=8,
        activation=Activation.RELU,
        projector_hidden_sizes=(4,),
        label_length=3,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: NonStationaryTransformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: NonStationaryTransformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: NonStationaryTransformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("is_causal", [False, True])
def test_de_stationary_attention_matches_reference(is_causal: bool) -> None:
    _ = torch.manual_seed(0)
    attention: DeStationaryAttention = DeStationaryAttention(8, 2, 0.0, is_causal).eval()
    queries: Tensor = torch.randn(3, 5, 8)
    keys: Tensor = torch.randn(3, 5, 8) if is_causal else torch.randn(3, 7, 8)
    scale: Tensor = torch.rand(3, 1) + 0.5
    shift: Tensor = torch.randn(3, keys.shape[1])
    with torch.no_grad():
        outputs: Tensor = attention(queries, keys, keys, scale, shift)
        projected_queries: Tensor = attention.query_projection(queries).reshape(3, 5, 2, 4)
        projected_keys: Tensor = attention.key_projection(keys).reshape(3, -1, 2, 4)
        projected_values: Tensor = attention.value_projection(keys).reshape(3, -1, 2, 4)
        scores: Tensor = torch.einsum("blhe,bshe->bhls", projected_queries, projected_keys)
        scores = scores * scale.reshape(3, 1, 1, 1) + shift.reshape(3, 1, 1, -1)
        if is_causal:
            is_future: Tensor = torch.ones(5, 5, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(is_future, float("-inf"))
        weights: Tensor = torch.softmax(scores / 2.0, dim=-1)
        mixed: Tensor = torch.einsum("bhls,bshd->blhd", weights, projected_values).flatten(start_dim=2)
        expected: Tensor = attention.output_projection(mixed)
    assert torch.allclose(outputs, expected, atol=1e-5)


def test_causal_attention_without_shift_ignores_future_keys() -> None:
    _ = torch.manual_seed(0)
    attention: DeStationaryAttention = DeStationaryAttention(4, 1, 0.0, is_causal=True).eval()
    tokens: Tensor = torch.randn(2, 6, 4)
    changed: Tensor = tokens.clone()
    changed[:, 4:] = torch.randn(2, 2, 4)
    scale: Tensor = torch.ones(2, 1)
    with torch.no_grad():
        original: Tensor = attention(tokens, tokens, tokens, scale, None)
        perturbed: Tensor = attention(changed, changed, changed, scale, None)
    assert torch.allclose(original[:, :4], perturbed[:, :4], atol=1e-6)


def test_stationarization_round_trip() -> None:
    values: Tensor = torch.randn(2, 9, 3) * 4.0 + 7.0
    stationarization: Stationarization = Stationarization.from_values(values)
    normalized: Tensor = stationarization.normalize(values)
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(normalized.std(dim=1, unbiased=False), torch.ones(2, 3), atol=1e-4)
    assert torch.allclose(stationarization.restore(normalized), values, atol=1e-5)


@pytest.mark.parametrize("channels", [1, 4])
def test_factor_learner_shapes(channels: int) -> None:
    learner: DeStationaryFactorLearner = DeStationaryFactorLearner(11, channels, (5, 3))
    values: Tensor = torch.randn(2, 11, channels)
    factors = learner(values, Stationarization.from_values(values))
    assert factors.scale.shape == (2, 1)
    assert bool((factors.scale > 0).all())
    assert factors.shift.shape == (2, 11)


def test_forecast_is_affine_equivariant_with_neutral_factors() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(
        input_length=12, input_channels=3, horizon=4, target_channels=(2, 0), num_parameters=2
    )
    network = MODELS[0].build_forecast_network(shape).eval()
    for module in network.modules():
        if isinstance(module, NonStationaryFactorProjector):
            last: nn.Module = module.layers[-1]
            assert isinstance(last, nn.Linear)
            _ = nn.init.zeros_(last.weight)
    past: Tensor = torch.randn(2, 12, 3)
    multiplier: Tensor = torch.tensor([2.0, 0.5, 3.0])
    offset: Tensor = torch.tensor([1.0, -4.0, 10.0])
    with torch.no_grad():
        original: Tensor = network(past)
        transformed: Tensor = network(past * multiplier + offset)
    targets: list[int] = list(shape.target_channels)
    expected_level: Tensor = original[..., 0] * multiplier[targets] + offset[targets]
    assert torch.allclose(transformed[..., 0], expected_level, atol=1e-3)
    assert torch.allclose(transformed[..., 1], original[..., 1], atol=1e-3)


def test_label_length_must_fit_the_input() -> None:
    shape: ForecastShape = ForecastShape(input_length=4, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="label_length"):
        NonStationaryTransformerParameters(model_size=4, num_heads=1, label_length=5).build_forecast_network(shape)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_encoder_layers"):
        NonStationaryTransformerParameters(num_encoder_layers=0)
    with pytest.raises(ValueError, match="num_heads"):
        NonStationaryTransformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="projector_hidden_sizes"):
        NonStationaryTransformerParameters(projector_hidden_sizes=())
    with pytest.raises(ValueError, match="projector_hidden_sizes"):
        NonStationaryTransformerParameters(projector_hidden_sizes=(4, 0))
    with pytest.raises(ValueError, match="dropout"):
        NonStationaryTransformerParameters(dropout=1.0)
