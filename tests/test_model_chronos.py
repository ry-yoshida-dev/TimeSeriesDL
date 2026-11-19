from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor

from time_series_dl.models.architectures.foundation.chronos import (
    ChronosForecastNetwork,
    ChronosOfficialNames,
    ChronosParameters,
    ChronosSize,
    ChronosTokenizer,
)
from time_series_dl.models.components.pretrained import (
    PretrainedWeights,
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)
from time_series_dl.models.interfaces import ForecastShape
from time_series_dl.models.interfaces.protocols.generative_forecasting import GenerativeForecastingModel
from time_series_dl.models.interfaces.protocols.zero_shot_forecasting import ZeroShotForecastingModel

TINY: ChronosParameters = ChronosParameters(
    model_size=16,
    key_value_size=4,
    num_heads=2,
    feedforward_size=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dropout=0.0,
    num_tokens=64,
    context_length=12,
    prediction_length=4,
)
RELATIVE_BIAS: str = "block.0.layer.0.SelfAttention.relative_attention_bias.weight"
SHAPE: ForecastShape = ForecastShape(
    input_length=10, input_channels=3, horizon=6, target_channels=(2, 0), num_parameters=5, level_parameters=(0,)
)


def build(parameters: ChronosParameters = TINY, shape: ForecastShape = SHAPE) -> ChronosForecastNetwork:
    _ = torch.manual_seed(0)
    network = parameters.build_generative_forecast_network(shape)
    assert isinstance(network, ChronosForecastNetwork)
    return network


def official_names(parameters: ChronosParameters) -> dict[str, tuple[int, ...]]:
    """Names and shapes of a T5ForConditionalGeneration checkpoint, written from the transformers naming."""
    size: int = parameters.model_size
    inner: int = parameters.num_heads * parameters.key_value_size
    feedforward: int = parameters.feedforward_size
    names: dict[str, tuple[int, ...]] = {"shared.weight": (parameters.num_tokens, size)}
    for stack, num_layers in (("encoder", parameters.num_encoder_layers), ("decoder", parameters.num_decoder_layers)):
        names[f"{stack}.final_layer_norm.weight"] = (size,)
        names[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = (
            parameters.relative_attention_num_buckets,
            parameters.num_heads,
        )
        for block in range(num_layers):
            attentions: list[tuple[int, str]] = [(0, "SelfAttention")]
            if stack == "decoder":
                attentions.append((1, "EncDecAttention"))
            for layer, attention in attentions:
                prefix: str = f"{stack}.block.{block}.layer.{layer}"
                names[f"{prefix}.layer_norm.weight"] = (size,)
                for projection in ("q", "k", "v"):
                    names[f"{prefix}.{attention}.{projection}.weight"] = (inner, size)
                names[f"{prefix}.{attention}.o.weight"] = (size, inner)
            dense: str = f"{stack}.block.{block}.layer.{len(attentions)}"
            names[f"{dense}.layer_norm.weight"] = (size,)
            names[f"{dense}.DenseReluDense.wi.weight"] = (feedforward, size)
            names[f"{dense}.DenseReluDense.wo.weight"] = (size, feedforward)
    return names


def random_official_weights(parameters: ChronosParameters) -> dict[str, Tensor]:
    generator: torch.Generator = torch.Generator().manual_seed(1)
    return {name: torch.randn(shape, generator=generator) for name, shape in official_names(parameters).items()}


def test_generative_network_shapes_and_gradients() -> None:
    network = build()
    past: Tensor = torch.randn(3, SHAPE.input_length, SHAPE.input_channels)
    future: Tensor = torch.randn(3, SHAPE.horizon, SHAPE.output_channels)
    _ = network.train()
    loss: Tensor = network.compute_loss(past, future)
    loss.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    outputs: Tensor = network(past)
    assert outputs.shape == (3, SHAPE.horizon, SHAPE.output_channels, SHAPE.num_parameters)
    assert bool(torch.isfinite(outputs).all())
    assert network.sample(past, 2).shape == (3, 2, SHAPE.horizon, SHAPE.output_channels)


def test_samples_are_reproducible_with_a_generator() -> None:
    network = build().eval()
    past: Tensor = torch.randn(2, SHAPE.input_length, SHAPE.input_channels)
    first: Tensor = network.sample(past, 3, torch.Generator().manual_seed(5))
    second: Tensor = network.sample(past, 3, torch.Generator().manual_seed(5))
    assert torch.equal(first, second)


def test_target_channels_are_forecast_separately() -> None:
    network = build(replace(TINY, top_k=1)).eval()
    past: Tensor = torch.randn(2, SHAPE.input_length, SHAPE.input_channels)
    samples: Tensor = network.sample(past, 1)
    alone: Tensor = network.model.sample(past[:, :, 2], SHAPE.horizon, 1, None)
    torch.testing.assert_close(samples[:, :, :, 0], alone)


def test_tokenizer_matches_explicit_quantization() -> None:
    tokenizer = ChronosTokenizer(num_tokens=12, low_limit=-3.0, high_limit=3.0)
    centers: Tensor = torch.linspace(-3.0, 3.0, 9)
    context: Tensor = torch.tensor([[2.0, -1.0, float("nan"), 3.0], [0.0, 0.0, 0.0, 0.0]])
    scale: Tensor = tokenizer.compute_scale(context)
    torch.testing.assert_close(scale, torch.tensor([2.0, 1.0]))
    values: Tensor = torch.tensor([[0.3, -100.0, 5.9, float("nan"), 100.0], [0.6, -0.6, 2.2, 0.0, -2.9]])
    tokens, is_observed = tokenizer.encode(values, scale)
    scaled: Tensor = values / scale.unsqueeze(-1)
    nearest: Tensor = (scaled.unsqueeze(-1) - centers).abs().argmin(dim=-1)
    expected: Tensor = torch.where(is_observed, nearest + 3, 0)
    assert torch.equal(tokens, expected)
    assert is_observed.tolist() == [[True, True, True, False, True], [True] * 5]
    decoded: Tensor = tokenizer.decode(tokens, scale)
    torch.testing.assert_close(decoded[1], centers[nearest[1]] * 1.0)
    torch.testing.assert_close(decoded[0, :3], centers[nearest[0, :3]] * 2.0)


def test_loss_is_teacher_forced_cross_entropy() -> None:
    network = build().eval()
    model = network.model
    context: Tensor = torch.randn(2, 9) + 2.0
    future: Tensor = torch.randn(2, 4) + 2.0
    future[0, 1] = float("nan")
    states, is_observed, scale = model.encode(context)
    labels, is_label_observed = model.tokenizer.encode(future, scale)
    decoder_inputs: Tensor = torch.cat([torch.zeros(2, 1, dtype=torch.long), labels], dim=1)
    logits: Tensor = model.compute_logits(model.decoder(model.token_embedding(decoder_inputs), states, is_observed))
    targets: Tensor = torch.cat([labels, torch.ones(2, 1, dtype=torch.long)], dim=1)
    is_counted: Tensor = torch.cat([is_label_observed, torch.ones(2, 1, dtype=torch.bool)], dim=1)
    expected: Tensor = functional.cross_entropy(logits[is_counted], targets[is_counted])
    torch.testing.assert_close(model.compute_loss(context, future), expected)


def test_greedy_sampling_matches_full_decoding() -> None:
    parameters: ChronosParameters = replace(TINY, top_k=1)
    network = build(parameters).eval()
    model = network.model
    context: Tensor = torch.randn(2, 15)
    sampled: Tensor = model.sample(context, 3, 1, None)
    states, is_observed, scale = model.encode(context)
    tokens: Tensor = torch.zeros(2, 1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(3):
            logits: Tensor = model.compute_logits(model.decoder(model.token_embedding(tokens), states, is_observed))
            logits[:, -1, 1] = -torch.inf
            tokens = torch.cat([tokens, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    torch.testing.assert_close(sampled[:, 0], model.tokenizer.decode(tokens[:, 1:], scale))


def test_horizon_beyond_prediction_length_continues_with_the_median() -> None:
    parameters: ChronosParameters = replace(TINY, top_k=1)
    network = build(parameters).eval()
    model = network.model
    context: Tensor = torch.randn(2, 10)
    samples: Tensor = model.sample(context, 7, 3, None)
    assert samples.shape == (2, 3, 7)
    first: Tensor = model.sample(context, 4, 3, None)
    torch.testing.assert_close(samples[..., :4], first)
    continued: Tensor = model.sample(torch.cat([context, first.median(dim=1).values], dim=1), 3, 3, None)
    torch.testing.assert_close(samples[..., 4:], continued)


def test_official_names_load(tmp_path: Path) -> None:
    tensors: dict[str, Tensor] = random_official_weights(TINY)
    tensors["lm_head.weight"] = tensors["shared.weight"].clone()
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, tensors)
    network = build()
    PretrainedWeightsLoader.load_file(network, path, ChronosOfficialNames.mapping())
    state: dict[str, Tensor] = network.state_dict()
    pairs: dict[str, str] = {
        "model.token_embedding.weight": "shared.weight",
        "model.encoder.position_bias.embedding.weight": f"encoder.{RELATIVE_BIAS}",
        "model.decoder.position_bias.embedding.weight": f"decoder.{RELATIVE_BIAS}",
        "model.encoder.blocks.1.self_attention.query.weight": "encoder.block.1.layer.0.SelfAttention.q.weight",
        "model.encoder.blocks.1.feed_forward.input.weight": "encoder.block.1.layer.1.DenseReluDense.wi.weight",
        "model.encoder.blocks.0.feed_forward_norm.weight": "encoder.block.0.layer.1.layer_norm.weight",
        "model.decoder.blocks.1.cross_attention.key.weight": "decoder.block.1.layer.1.EncDecAttention.k.weight",
        "model.decoder.blocks.0.cross_attention_norm.weight": "decoder.block.0.layer.1.layer_norm.weight",
        "model.decoder.blocks.1.feed_forward.output.weight": "decoder.block.1.layer.2.DenseReluDense.wo.weight",
        "model.decoder.final_norm.weight": "decoder.final_layer_norm.weight",
    }
    for ours, published in pairs.items():
        assert torch.equal(state[ours], tensors[published]), ours


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
def test_mismatched_official_weights_raise(change: str) -> None:
    tensors: dict[str, Tensor] = random_official_weights(TINY)
    match change:
        case "missing":
            del tensors["decoder.final_layer_norm.weight"]
        case "unexpected":
            tensors["decoder.block.0.layer.3.layer_norm.weight"] = torch.ones(16)
        case _:
            tensors["shared.weight"] = torch.ones(65, 16)
    with pytest.raises(WeightsMismatchError):
        PretrainedWeightsLoader.load_tensors(build(), tensors, ChronosOfficialNames.mapping())


def test_pretrained_parameters() -> None:
    parameters: ChronosParameters = ChronosParameters.pretrained(ChronosSize.TINY)
    assert parameters.weights == PretrainedWeights(
        repository="amazon/chronos-t5-tiny", revision="29d808298f1a62493e7b9a5e08529d0d930fa189"
    )
    assert parameters.is_pretrained
    assert not parameters.without_pretrained_weights().is_pretrained
    assert parameters.without_pretrained_weights().model_size == 256
    assert ChronosParameters.pretrained(ChronosSize.SMALL).without_pretrained_weights() == ChronosParameters()
    assert isinstance(parameters, GenerativeForecastingModel)
    assert isinstance(parameters, ZeroShotForecastingModel)
    assert parameters.zero_shot_quantile_levels == ()


@pytest.mark.parametrize(
    "create",
    [
        lambda: ChronosParameters(num_tokens=4),
        lambda: ChronosParameters(low_limit=1.0, high_limit=1.0),
        lambda: ChronosParameters(temperature=0.0),
        lambda: ChronosParameters(top_p=0.0),
        lambda: ChronosParameters(top_k=0),
        lambda: ChronosParameters(context_length=0),
        lambda: ChronosParameters(dropout=1.0),
        lambda: ChronosParameters(model_size=0),
    ],
)
def test_invalid_parameters_raise(create: Callable[[], ChronosParameters]) -> None:
    with pytest.raises(ValueError):
        _ = create()
