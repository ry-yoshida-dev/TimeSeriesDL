from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.foundation.timer import (
    TimerDecoderLayer,
    TimerForecastNetwork,
    TimerOfficialNames,
    TimerParameters,
    TimerRotaryEmbedding,
    TimerSize,
)
from time_series_dl.models.components.pretrained import (
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)

MODEL: TimerParameters = TimerParameters(
    patch_length=4, output_patch_length=4, model_size=16, feedforward_size=24, num_layers=2, num_heads=2
)


def build_network(input_length: int, horizon: int, model: TimerParameters = MODEL) -> TimerForecastNetwork:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(
        input_length=input_length, input_channels=1, horizon=horizon, target_channels=(0,)
    )
    network = model.build_forecast_network(shape)
    assert isinstance(network, TimerForecastNetwork)
    return network.eval()


def official_state_dict(model: TimerParameters) -> dict[str, Tensor]:
    size: int = model.model_size
    feedforward: int = model.feedforward_size
    tensors: dict[str, Tensor] = {
        "model.embed_layer.emb.weight": torch.randn(size, model.patch_length),
        "model.norm.weight": torch.randn(size),
        "model.norm.bias": torch.randn(size),
        "lm_heads.0.weight": torch.randn(model.output_patch_length, size),
    }
    for layer in range(model.num_layers):
        prefix: str = f"model.layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj"):
            tensors[f"{prefix}self_attn.{projection}.weight"] = torch.randn(size, size)
            tensors[f"{prefix}self_attn.{projection}.bias"] = torch.randn(size)
        tensors[f"{prefix}self_attn.o_proj.weight"] = torch.randn(size, size)
        tensors[f"{prefix}ffn_layer.gate_proj.weight"] = torch.randn(feedforward, size)
        tensors[f"{prefix}ffn_layer.up_proj.weight"] = torch.randn(feedforward, size)
        tensors[f"{prefix}ffn_layer.down_proj.weight"] = torch.randn(size, feedforward)
        for norm in ("norm1", "norm2"):
            tensors[f"{prefix}{norm}.weight"] = torch.randn(size)
            tensors[f"{prefix}{norm}.bias"] = torch.randn(size)
    return tensors


@pytest.mark.parametrize(
    "shape",
    [
        ForecastShape(input_length=12, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
        ForecastShape(input_length=10, input_channels=4, horizon=3, target_channels=(2, 0)),
        ForecastShape(input_length=4, input_channels=1, horizon=9, target_channels=(0,)),
    ],
)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(MODEL, shape)


def test_rejects_quantiles_and_short_windows() -> None:
    with pytest.raises(ValueError, match="num_parameters"):
        _ = MODEL.build_forecast_network(
            ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,), num_parameters=3)
        )
    with pytest.raises(ValueError, match="one patch"):
        _ = MODEL.build_forecast_network(
            ForecastShape(input_length=3, input_channels=1, horizon=2, target_channels=(0,))
        )


def test_generation_normalizes_each_block_by_its_own_statistics() -> None:
    network: TimerForecastNetwork = build_network(input_length=10, horizon=6)
    series: Tensor = torch.randn(2, 10) * 3.0 + 5.0
    with torch.no_grad():
        forecast: Tensor = network.forecast(series)
        context: Tensor = series[:, -8:]
        mean: Tensor = context.mean(dim=-1, keepdim=True)
        std: Tensor = context.std(dim=-1, keepdim=True)
        first_patches: Tensor = ((context - mean) / std).reshape(2, 2, 4)
        first: Tensor = network.model(first_patches)[:, -1] * std + mean
        block_mean: Tensor = first.mean(dim=-1, keepdim=True)
        block_std: Tensor = first.std(dim=-1, keepdim=True)
        patches: Tensor = torch.cat([first_patches, ((first - block_mean) / block_std).unsqueeze(1)], dim=1)
        second: Tensor = network.model(patches)[:, -1] * block_std + block_mean
    torch.testing.assert_close(forecast, torch.cat([first, second[:, :2]], dim=-1))


def test_forecast_is_equivariant_to_affine_maps() -> None:
    network: TimerForecastNetwork = build_network(input_length=12, horizon=7)
    series: Tensor = torch.randn(3, 12)
    with torch.no_grad():
        forecast: Tensor = network.forecast(series)
        transformed: Tensor = network.forecast(series * 4.0 - 2.0)
    torch.testing.assert_close(transformed, forecast * 4.0 - 2.0, atol=1e-4, rtol=1e-4)


def test_window_is_cut_to_whole_patches() -> None:
    long: TimerForecastNetwork = build_network(input_length=11, horizon=3)
    short: TimerForecastNetwork = build_network(input_length=8, horizon=3)
    past: Tensor = torch.randn(2, 11, 1)
    with torch.no_grad():
        torch.testing.assert_close(long(past), short(past[:, -8:]))


def test_rotary_embedding_depends_on_relative_positions() -> None:
    embedding: TimerRotaryEmbedding = TimerRotaryEmbedding(size=8, base=10000.0)
    query: Tensor = torch.randn(8).expand(1, 1, 6, 8)
    key: Tensor = torch.randn(8).expand(1, 1, 6, 8)
    scores: Tensor = embedding(query)[0, 0] @ embedding(key)[0, 0].T
    torch.testing.assert_close(scores[1, 3], scores[2, 4])
    torch.testing.assert_close(scores[0, 0], query[0, 0, 0] @ key[0, 0, 0])
    angle: Tensor = torch.tensor(1.0)
    rotated: Tensor = embedding(torch.eye(8)[[0]].expand(1, 1, 2, 8))[0, 0, 1]
    torch.testing.assert_close(rotated[[0, 4]], torch.stack([angle.cos(), angle.sin()]))


def test_official_names_load(tmp_path: Path) -> None:
    tensors: dict[str, Tensor] = official_state_dict(MODEL)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, tensors)
    network: TimerForecastNetwork = build_network(input_length=8, horizon=4)
    PretrainedWeightsLoader.load_file(network, path, TimerOfficialNames.mapping())
    model = network.model
    torch.testing.assert_close(model.patch_embedding.weight, tensors["model.embed_layer.emb.weight"])
    torch.testing.assert_close(model.head.weight, tensors["lm_heads.0.weight"])
    torch.testing.assert_close(model.final_norm.bias, tensors["model.norm.bias"])
    layer = model.layers[1]
    assert isinstance(layer, TimerDecoderLayer)
    torch.testing.assert_close(layer.attention.query.bias, tensors["model.layers.1.self_attn.q_proj.bias"])
    torch.testing.assert_close(layer.attention.output.weight, tensors["model.layers.1.self_attn.o_proj.weight"])
    torch.testing.assert_close(layer.feed_forward.gate.weight, tensors["model.layers.1.ffn_layer.gate_proj.weight"])
    torch.testing.assert_close(layer.feed_forward.input.weight, tensors["model.layers.1.ffn_layer.up_proj.weight"])
    torch.testing.assert_close(layer.attention_norm.weight, tensors["model.layers.1.norm1.weight"])
    torch.testing.assert_close(layer.feed_forward_norm.weight, tensors["model.layers.1.norm2.weight"])


def test_official_names_are_strict() -> None:
    network: TimerForecastNetwork = build_network(input_length=8, horizon=4)
    missing: dict[str, Tensor] = official_state_dict(MODEL)
    del missing["model.norm.bias"]
    with pytest.raises(WeightsMismatchError, match=r"missing: model\.final_norm\.bias"):
        PretrainedWeightsLoader.load_tensors(network, missing, TimerOfficialNames.mapping())
    unexpected: dict[str, Tensor] = {**official_state_dict(MODEL), "lm_heads.1.weight": torch.randn(8, 16)}
    with pytest.raises(WeightsMismatchError, match="unexpected: lm_heads.1.weight"):
        PretrainedWeightsLoader.load_tensors(network, unexpected, TimerOfficialNames.mapping())
    mismatched: dict[str, Tensor] = {**official_state_dict(MODEL), "lm_heads.0.weight": torch.randn(8, 16)}
    with pytest.raises(WeightsMismatchError, match="shape mismatch: model.head.weight"):
        PretrainedWeightsLoader.load_tensors(network, mismatched, TimerOfficialNames.mapping())


def test_pretrained_parameters() -> None:
    parameters: TimerParameters = TimerParameters.pretrained(TimerSize.BASE)
    assert parameters.is_pretrained
    assert parameters.weights is not None
    assert parameters.weights.repository == "thuml/timer-base-84m"
    assert parameters.without_pretrained_weights() == TimerParameters()
    assert not TimerParameters().is_pretrained
    assert parameters.zero_shot_quantile_levels == ()


INVALID_PARAMETERS: list[Callable[[], TimerParameters]] = [
    lambda: TimerParameters(patch_length=0),
    lambda: TimerParameters(num_heads=3),
    lambda: TimerParameters(model_size=12, num_heads=4),
    lambda: TimerParameters(output_patch_length=6),
    lambda: TimerParameters(attention_dropout=1.0),
    lambda: TimerParameters(rotary_base=1.0),
]


@pytest.mark.parametrize("build", INVALID_PARAMETERS)
def test_invalid_parameters(build: Callable[[], TimerParameters]) -> None:
    with pytest.raises(ValueError):
        _ = build()
