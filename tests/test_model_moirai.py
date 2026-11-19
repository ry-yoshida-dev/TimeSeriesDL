import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.distributions import Categorical, LogNormal, Normal, StudentT

from time_series_dl.models.architectures.foundation.moirai import (
    MoiraiEncoderLayer,
    MoiraiForecastNetwork,
    MoiraiMixture,
    MoiraiModel,
    MoiraiOfficialNames,
    MoiraiParameters,
    MoiraiRotaryEmbedding,
    MoiraiSize,
)
from time_series_dl.models.components.pretrained import (
    PretrainedWeights,
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)
from time_series_dl.models.interfaces import ForecastShape
from time_series_dl.models.interfaces.generative_forecast_network import GenerativeForecastNetwork
from time_series_dl.models.interfaces.protocols.generative_forecasting import GenerativeForecastingModel
from time_series_dl.models.interfaces.protocols.zero_shot_forecasting import ZeroShotForecastingModel

TINY: MoiraiParameters = MoiraiParameters(model_size=64, num_layers=2, patch_sizes=(4, 8), patch_size=4)
SHAPE: ForecastShape = ForecastShape(
    input_length=18, input_channels=3, horizon=6, target_channels=(2, 0), num_parameters=5, level_parameters=(0,)
)


def build(parameters: MoiraiParameters = TINY, shape: ForecastShape = SHAPE) -> MoiraiForecastNetwork:
    _ = torch.manual_seed(0)
    network = parameters.build_generative_forecast_network(shape)
    assert isinstance(network, MoiraiForecastNetwork)
    return network


def official_state_dict(parameters: MoiraiParameters) -> dict[str, Tensor]:
    """Names and shapes of uni2ts' ``MoiraiModule``, written out independently of the mapping."""
    size: int = parameters.model_size
    sizes: int = len(parameters.patch_sizes)
    widest: int = max(parameters.patch_sizes)
    heads: int = size // 64
    hidden: int = (int(4 * size * 2 / 3) + 7) // 8 * 8
    shapes: dict[str, tuple[int, ...]] = {
        "mask_encoding.weight": (1, size),
        "in_proj.weight": (sizes, size, widest),
        "in_proj.bias": (sizes, size),
        "encoder.norm.weight": (size,),
        "param_proj.proj.weights_logits.weight": (sizes, 4 * widest, size),
        "param_proj.proj.weights_logits.bias": (sizes, 4 * widest),
    }
    for component in ("0.df", "0.loc", "0.scale", "1.loc", "2.total_count", "2.logits", "3.loc", "3.scale"):
        shapes[f"param_proj.proj.components.{component}.weight"] = (sizes, widest, size)
        shapes[f"param_proj.proj.components.{component}.bias"] = (sizes, widest)
    for layer in range(parameters.num_layers):
        prefix: str = f"encoder.layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[prefix + f"self_attn.{projection}.weight"] = (size, size)
        shapes[prefix + "self_attn.q_norm.weight"] = (64,)
        shapes[prefix + "self_attn.k_norm.weight"] = (64,)
        shapes[prefix + "self_attn.var_attn_bias.emb.weight"] = (2, heads)
        shapes[prefix + "ffn.fc1.weight"] = (hidden, size)
        shapes[prefix + "ffn.fc_gate.weight"] = (hidden, size)
        shapes[prefix + "ffn.fc2.weight"] = (size, hidden)
        shapes[prefix + "norm1.weight"] = (size,)
        shapes[prefix + "norm2.weight"] = (size,)
    generator = torch.Generator().manual_seed(1)
    return {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}


def test_samples_have_the_forecast_shape() -> None:
    network = build()
    past: Tensor = torch.randn(2, SHAPE.input_length, SHAPE.input_channels)
    _ = network.eval()
    samples: Tensor = network(past)
    assert samples.shape == (2, SHAPE.horizon, SHAPE.output_channels, SHAPE.num_parameters)
    assert bool(torch.isfinite(samples).all())
    assert network.sample(past, 3).shape == (2, 3, SHAPE.horizon, SHAPE.output_channels)


@pytest.mark.parametrize("patch_size", [4, None])
def test_every_parameter_receives_a_gradient_from_the_loss(patch_size: int | None) -> None:
    network = build(MoiraiParameters(model_size=64, num_layers=1, patch_sizes=(4, 8), patch_size=patch_size))
    _ = network.train()
    loss: Tensor = network.compute_loss(
        torch.randn(2, SHAPE.input_length, SHAPE.input_channels), torch.randn(2, SHAPE.horizon, 2)
    )
    assert loss.dim() == 0
    loss.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_automatic_patch_size_selects_per_window() -> None:
    network = build(MoiraiParameters(model_size=64, num_layers=1, patch_sizes=(4, 8)))
    past: Tensor = torch.randn(3, SHAPE.input_length, SHAPE.input_channels)
    selected: Tensor = network._select_patch_sizes(past)  # pyright: ignore[reportPrivateUsage]
    assert selected.shape == (3,)
    assert bool(((selected >= 0) & (selected < 2)).all())
    _ = network.eval()
    assert network.sample(past, 4).shape == (3, 4, SHAPE.horizon, 2)


def test_automatic_patch_size_needs_a_window_longer_than_the_horizon() -> None:
    shape = ForecastShape(input_length=6, input_channels=1, horizon=6, target_channels=(0,))
    with pytest.raises(ValueError, match="input_length > horizon"):
        MoiraiParameters(model_size=64, num_layers=1).build_forecast_network(shape)


def test_generator_makes_samples_reproducible_without_touching_the_global_generator() -> None:
    network = build()
    past: Tensor = torch.randn(2, SHAPE.input_length, SHAPE.input_channels)
    state: Tensor = torch.get_rng_state()
    first: Tensor = network.sample(past, 4, torch.Generator().manual_seed(3))
    second: Tensor = network.sample(past, 4, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert torch.equal(state, torch.get_rng_state())


def test_covariate_channels_inform_the_forecast() -> None:
    network = build()
    past: Tensor = torch.randn(1, SHAPE.input_length, SHAPE.input_channels)
    changed: Tensor = past.clone()
    changed[..., 1] = torch.randn(SHAPE.input_length)
    future: Tensor = torch.randn(1, SHAPE.horizon, 2)
    with torch.no_grad():
        assert not torch.allclose(network.compute_loss(past, future), network.compute_loss(changed, future))


def test_statistics_are_the_unbiased_standardization_of_observed_past_values() -> None:
    model = MoiraiModel(TINY)
    values: Tensor = torch.tensor([[[1.0, 2.0, 4.0, 0.0], [3.0, 3.0, 3.0, 3.0]]])
    is_observed: Tensor = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
    location, scale = model._compute_statistics(values, is_observed)  # pyright: ignore[reportPrivateUsage]
    mean: float = 7.0 / 3.0
    variance: float = ((1 - mean) ** 2 + (2 - mean) ** 2 + (4 - mean) ** 2) / 2
    torch.testing.assert_close(location[0, :, 0, 0], torch.tensor([mean, 3.0]))
    torch.testing.assert_close(scale[0, :, 0, 0], torch.tensor([math.sqrt(variance + 1e-5), math.sqrt(1e-5)]))


def test_time_ids_count_patches_from_the_first_observed_target() -> None:
    is_observed: Tensor = torch.zeros(1, 2, 5, 4, dtype=torch.bool)
    is_observed[0, 1, 2:] = True
    time_ids: Tensor = MoiraiModel._compute_time_ids(is_observed)  # pyright: ignore[reportPrivateUsage]
    assert time_ids.tolist() == [[0, 0, 0, 1, 2]]


def test_missing_values_are_masked_out_of_the_statistics() -> None:
    network = build()
    past: Tensor = torch.randn(1, SHAPE.input_length, SHAPE.input_channels)
    with_gap: Tensor = past.clone()
    with_gap[0, 3, 0] = float("nan")
    targets: Tensor = with_gap[..., [2, 0]].transpose(1, 2)
    distribution = network.model(targets, with_gap[..., [1]].transpose(1, 2), SHAPE.horizon, 4)
    observed: Tensor = torch.cat([past[0, :3, 0], past[0, 4:, 0]])
    torch.testing.assert_close(distribution.location[0, 1, 0, 0], observed.mean())


def test_rotary_embedding_rotates_adjacent_pairs_of_the_first_half() -> None:
    rotary = MoiraiRotaryEmbedding(8)
    values: Tensor = torch.randn(1, 1, 3, 8)
    positions: Tensor = torch.tensor([[0, 2, 5]])
    rotated: Tensor = rotary.apply(values, positions)
    for token, position in enumerate((0, 2, 5)):
        for pair, frequency in enumerate((1.0, 1.0 / 100.0)):
            angle: float = position * frequency
            first, second = values[0, 0, token, 2 * pair].item(), values[0, 0, token, 2 * pair + 1].item()
            expected: Tensor = torch.tensor(
                [first * math.cos(angle) - second * math.sin(angle), second * math.cos(angle) + first * math.sin(angle)]
            )
            torch.testing.assert_close(rotated[0, 0, token, 2 * pair : 2 * pair + 2], expected)
    torch.testing.assert_close(rotated[..., 4:], values[..., 4:])


def test_mixture_log_density_matches_an_explicit_mixture() -> None:
    generator = torch.Generator().manual_seed(0)
    shape: tuple[int, ...] = (6,)
    mixture = MoiraiMixture(
        mixture_logits=torch.randn(*shape, 4, generator=generator),
        student_t_degrees_of_freedom=2.0 + torch.rand(shape, generator=generator) * 5,
        student_t_location=torch.randn(shape, generator=generator),
        student_t_scale=torch.rand(shape, generator=generator) + 0.5,
        normal_location=torch.randn(shape, generator=generator),
        normal_scale=0.5,
        negative_binomial_total_count=torch.rand(shape, generator=generator) + 0.5,
        negative_binomial_logits=torch.randn(shape, generator=generator),
        log_normal_location=torch.randn(shape, generator=generator),
        log_normal_scale=torch.rand(shape, generator=generator) + 0.5,
    )
    values: Tensor = torch.tensor([-1.5, 0.0, 0.5, 2.0, 3.0, 7.0])
    weights: Tensor = Categorical(logits=mixture.mixture_logits).probs
    count: Tensor = mixture.negative_binomial_total_count
    probability: Tensor = torch.sigmoid(mixture.negative_binomial_logits)
    negative_binomial: Tensor = torch.exp(
        torch.lgamma(values.clamp(min=0) + count)
        - torch.lgamma(count)
        - torch.lgamma(values.clamp(min=0) + 1)
        + count * torch.log1p(-probability)
        + values.clamp(min=0) * torch.log(probability)
    )
    densities: Tensor = torch.stack(
        [
            StudentT(mixture.student_t_degrees_of_freedom, mixture.student_t_location, mixture.student_t_scale)
            .log_prob(values)
            .exp(),
            Normal(mixture.normal_location, 0.5).log_prob(values).exp(),
            torch.where(values >= 0, negative_binomial, 0.0),
            torch.where(
                values > 0,
                LogNormal(mixture.log_normal_location, mixture.log_normal_scale).log_prob(values.clamp(min=1e-6)).exp(),
                0.0,
            ),
        ],
        dim=-1,
    )
    torch.testing.assert_close(mixture.log_prob(values), (weights * densities).sum(dim=-1).log())


def test_published_names_load_into_the_network(tmp_path: Path) -> None:
    network = build()
    published: dict[str, Tensor] = official_state_dict(TINY)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, published)
    PretrainedWeightsLoader.load_file(network, path, MoiraiOfficialNames.mapping())
    layer = network.model.encoder.layers[1]
    assert isinstance(layer, MoiraiEncoderLayer)
    torch.testing.assert_close(network.model.input_projection.weight, published["in_proj.weight"])
    torch.testing.assert_close(network.model.mask_embedding.weight, published["mask_encoding.weight"])
    torch.testing.assert_close(layer.attention.key.weight, published["encoder.layers.1.self_attn.k_proj.weight"])
    torch.testing.assert_close(
        layer.attention.variate_bias.weight, published["encoder.layers.1.self_attn.var_attn_bias.emb.weight"]
    )
    torch.testing.assert_close(layer.feed_forward.gate.weight, published["encoder.layers.1.ffn.fc_gate.weight"])
    torch.testing.assert_close(layer.feed_forward_norm.weight, published["encoder.layers.1.norm2.weight"])
    head = network.model.distribution_head
    torch.testing.assert_close(head.mixture_logits.weight, published["param_proj.proj.weights_logits.weight"])
    torch.testing.assert_close(
        head.negative_binomial_total_count.bias, published["param_proj.proj.components.2.total_count.bias"]
    )
    torch.testing.assert_close(head.log_normal_scale.weight, published["param_proj.proj.components.3.scale.weight"])


def test_published_weights_must_fit_exactly(tmp_path: Path) -> None:
    network = build()
    published: dict[str, Tensor] = official_state_dict(TINY)
    missing: dict[str, Tensor] = {name: tensor for name, tensor in published.items() if name != "encoder.norm.weight"}
    unexpected: dict[str, Tensor] = {**published, "encoder.extra.weight": torch.zeros(1)}
    reshaped: dict[str, Tensor] = {**published, "mask_encoding.weight": torch.zeros(1, 32)}
    for weights in (missing, unexpected, reshaped):
        with pytest.raises(WeightsMismatchError):
            PretrainedWeightsLoader.load_tensors(network, weights, MoiraiOfficialNames.mapping())


def test_parameters_follow_the_published_architecture() -> None:
    small = MoiraiParameters.pretrained(MoiraiSize.SMALL)
    assert small.num_heads == 6
    assert small.feedforward_size == 1024
    assert MoiraiParameters.pretrained(MoiraiSize.BASE).feedforward_size == 2048
    assert small.weights == PretrainedWeights(
        repository="Salesforce/moirai-1.1-R-small", revision="0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
    )
    assert small.is_pretrained
    assert not small.without_pretrained_weights().is_pretrained
    assert small.without_pretrained_weights() == MoiraiParameters()
    assert isinstance(small, ZeroShotForecastingModel)
    assert isinstance(small, GenerativeForecastingModel)
    assert small.zero_shot_quantile_levels == ()
    assert isinstance(TINY.build_forecast_network(SHAPE), GenerativeForecastNetwork)


@pytest.mark.parametrize(
    ("build_parameters", "message"),
    [
        (lambda: MoiraiParameters(model_size=100), "multiple of 64"),
        (lambda: MoiraiParameters(num_layers=0), "num_layers"),
        (lambda: MoiraiParameters(patch_sizes=(8, 8)), "strictly increasing"),
        (lambda: MoiraiParameters(patch_sizes=()), "patch_sizes"),
        (lambda: MoiraiParameters(patch_size=12), "patch_size"),
        (lambda: MoiraiParameters(normal_scale=0.0), "normal_scale"),
        (lambda: MoiraiParameters(dropout=1.0), "dropout"),
        (lambda: MoiraiParameters(layer_norm_epsilon=0.0), "layer_norm_epsilon"),
    ],
)
def test_invalid_parameters_raise(build_parameters: Callable[[], MoiraiParameters], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ = build_parameters()
