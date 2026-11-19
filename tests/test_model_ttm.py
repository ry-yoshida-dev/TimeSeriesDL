from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from network_checks import check_forecast_network
from torch import Tensor, nn

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.foundation.ttm import (
    TTMForecastNetwork,
    TTMMixerBlock,
    TTMMixerConfiguration,
    TTMMixerLayer,
    TTMOfficialNames,
    TTMParameters,
    TTMVariant,
)
from time_series_dl.models.components.pretrained import (
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)

MODEL: TTMParameters = TTMParameters(
    context_length=16,
    prediction_length=8,
    patch_length=4,
    patch_stride=4,
    model_size=16,
    num_layers=2,
    adaptive_patching_levels=2,
    decoder_model_size=8,
    decoder_num_layers=1,
)


def build_network(input_length: int, horizon: int, model: TTMParameters = MODEL) -> TTMForecastNetwork:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(
        input_length=input_length, input_channels=1, horizon=horizon, target_channels=(0,)
    )
    network = model.build_forecast_network(shape)
    assert isinstance(network, TTMForecastNetwork)
    return network.eval()


def add_mixer_layer(tensors: dict[str, Tensor], prefix: str, num_patches: int, size: int) -> None:
    for mixer, width in (("patch_mixer", num_patches), ("feature_mixer", size)):
        tensors[f"{prefix}{mixer}.norm.norm.weight"] = torch.randn(size)
        tensors[f"{prefix}{mixer}.norm.norm.bias"] = torch.randn(size)
        tensors[f"{prefix}{mixer}.mlp.fc1.weight"] = torch.randn(2 * width, width)
        tensors[f"{prefix}{mixer}.mlp.fc1.bias"] = torch.randn(2 * width)
        tensors[f"{prefix}{mixer}.mlp.fc2.weight"] = torch.randn(width, 2 * width)
        tensors[f"{prefix}{mixer}.mlp.fc2.bias"] = torch.randn(width)
        tensors[f"{prefix}{mixer}.gating_block.attn_layer.weight"] = torch.randn(width, width)
        tensors[f"{prefix}{mixer}.gating_block.attn_layer.bias"] = torch.randn(width)


def official_state_dict(model: TTMParameters) -> dict[str, Tensor]:
    num_patches: int = model.num_patches
    tensors: dict[str, Tensor] = {
        "backbone.encoder.patcher.weight": torch.randn(model.model_size, model.patch_length),
        "backbone.encoder.patcher.bias": torch.randn(model.model_size),
        "decoder.adapter.weight": torch.randn(model.decoder_model_size, model.model_size),
        "decoder.adapter.bias": torch.randn(model.decoder_model_size),
        "head.base_forecast_block.weight": torch.randn(model.prediction_length, num_patches * model.decoder_model_size),
        "head.base_forecast_block.bias": torch.randn(model.prediction_length),
    }
    for block, level in enumerate(reversed(range(model.adaptive_patching_levels))):
        for layer in range(model.num_layers):
            add_mixer_layer(
                tensors,
                f"backbone.encoder.mlp_mixer_encoder.mixers.{block}.mixer_layers.{layer}.",
                num_patches * 2**level,
                model.model_size // 2**level,
            )
    for layer in range(model.decoder_num_layers):
        add_mixer_layer(tensors, f"decoder.decoder_block.mixers.{layer}.", num_patches, model.decoder_model_size)
    return tensors


@pytest.mark.parametrize(
    "shape",
    [
        ForecastShape(input_length=16, input_channels=3, horizon=8, target_channels=(0, 1, 2)),
        ForecastShape(input_length=11, input_channels=4, horizon=5, target_channels=(2, 0)),
        ForecastShape(input_length=23, input_channels=1, horizon=1, target_channels=(0,)),
    ],
)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(MODEL, shape)


def test_overlapping_patches_without_decoder_adapter() -> None:
    model: TTMParameters = TTMParameters(
        context_length=15,
        prediction_length=4,
        patch_length=6,
        patch_stride=3,
        model_size=8,
        num_layers=1,
        adaptive_patching_levels=0,
        decoder_model_size=8,
        decoder_num_layers=1,
        is_gated_attention_used=False,
    )
    assert model.num_patches == 4
    check_forecast_network(model, ForecastShape(input_length=15, input_channels=2, horizon=4, target_channels=(1,)))
    assert build_network(15, 4, model).model.adapter is None


def test_rejects_quantiles_and_long_horizons() -> None:
    with pytest.raises(ValueError, match="num_parameters"):
        _ = MODEL.build_forecast_network(
            ForecastShape(input_length=16, input_channels=1, horizon=2, target_channels=(0,), num_parameters=3)
        )
    with pytest.raises(ValueError, match="prediction_length"):
        _ = MODEL.build_forecast_network(
            ForecastShape(input_length=16, input_channels=1, horizon=9, target_channels=(0,))
        )


def test_standardization_counts_left_padding() -> None:
    network: TTMForecastNetwork = build_network(input_length=12, horizon=8)
    head: nn.Linear = network.model.head
    with torch.no_grad():
        _ = head.weight.zero_()
        _ = head.bias.fill_(2.0)
        series: Tensor = torch.randn(3, 12) * 5.0 + 3.0
        padded: Tensor = torch.cat([torch.zeros(3, 4), series], dim=-1)
        location: Tensor = padded.mean(dim=-1, keepdim=True)
        scale: Tensor = (padded.var(dim=-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        torch.testing.assert_close(network.model(series), (2.0 * scale + location).expand(3, 8))
        torch.testing.assert_close(network.model(torch.cat([torch.randn(3, 5), padded], dim=-1)), network.model(series))


def test_adaptive_patching_splits_each_patch() -> None:
    configuration: TTMMixerConfiguration = MODEL.encoder_configuration
    block: TTMMixerBlock = TTMMixerBlock(configuration, patch_factor=2).eval()
    hidden: Tensor = torch.randn(2, 4, 16)
    split: Tensor = torch.cat([hidden[:, :, None, :8], hidden[:, :, None, 8:]], dim=2).reshape(2, 8, 8)
    for layer in block.layers:
        assert isinstance(layer, TTMMixerLayer)
        split = layer(split)
    torch.testing.assert_close(block(hidden), split.reshape(2, 4, 16))
    assert configuration.compute_patch_factor(1) == 2
    assert configuration.compute_patch_factor(2) == 1


def test_official_names_load(tmp_path: Path) -> None:
    tensors: dict[str, Tensor] = official_state_dict(MODEL)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, tensors)
    network: TTMForecastNetwork = build_network(input_length=16, horizon=8)
    PretrainedWeightsLoader.load_file(network, path, TTMOfficialNames.mapping(MODEL))
    state: dict[str, Tensor] = network.state_dict()
    for published, ours in (
        ("backbone.encoder.patcher.weight", "model.patch_embedding.weight"),
        ("decoder.adapter.bias", "model.adapter.bias"),
        ("head.base_forecast_block.weight", "model.head.weight"),
        (
            "backbone.encoder.mlp_mixer_encoder.mixers.0.mixer_layers.1.patch_mixer.mlp.fc1.weight",
            "model.encoder.blocks.0.layers.1.patch_mixer.perceptron.hidden.weight",
        ),
        (
            "backbone.encoder.mlp_mixer_encoder.mixers.1.mixer_layers.0.feature_mixer.gating_block.attn_layer.bias",
            "model.encoder.blocks.1.layers.0.feature_mixer.gate.projection.bias",
        ),
        (
            "decoder.decoder_block.mixers.0.patch_mixer.norm.norm.weight",
            "model.decoder.blocks.0.layers.0.patch_mixer.norm.weight",
        ),
        (
            "decoder.decoder_block.mixers.0.feature_mixer.mlp.fc2.weight",
            "model.decoder.blocks.0.layers.0.feature_mixer.perceptron.output.weight",
        ),
    ):
        torch.testing.assert_close(state[ours], tensors[published])


def test_official_names_are_strict() -> None:
    network: TTMForecastNetwork = build_network(input_length=16, horizon=8)
    mapping = TTMOfficialNames.mapping(MODEL)
    missing: dict[str, Tensor] = official_state_dict(MODEL)
    del missing["decoder.adapter.bias"]
    with pytest.raises(WeightsMismatchError, match=r"missing: model\.adapter\.bias"):
        PretrainedWeightsLoader.load_tensors(network, missing, mapping)
    unexpected: dict[str, Tensor] = {**official_state_dict(MODEL), "backbone.encoder.freq_mod.0.weight": torch.ones(1)}
    with pytest.raises(WeightsMismatchError, match="unexpected"):
        PretrainedWeightsLoader.load_tensors(network, unexpected, mapping)
    mismatched: dict[str, Tensor] = {**official_state_dict(MODEL), "head.base_forecast_block.bias": torch.ones(96)}
    with pytest.raises(WeightsMismatchError, match=r"shape mismatch: model\.head\.bias"):
        PretrainedWeightsLoader.load_tensors(network, mismatched, mapping)


@pytest.mark.parametrize("variant", list(TTMVariant))
def test_pretrained_variants(variant: TTMVariant) -> None:
    parameters: TTMParameters = TTMParameters.pretrained(variant)
    assert parameters.weights is not None
    assert parameters.weights.revision == variant.revision
    assert parameters.is_pretrained
    assert parameters.zero_shot_quantile_levels == ()
    assert parameters.prediction_length == variant.prediction_length
    assert parameters.num_patches == {512: 8, 1024: 8, 1536: 12}[variant.context_length]
    assert not parameters.without_pretrained_weights().is_pretrained


def test_main_variant_has_the_default_architecture() -> None:
    parameters: TTMParameters = TTMParameters.pretrained(TTMVariant.CONTEXT_512_PREDICTION_96)
    assert parameters.without_pretrained_weights() == TTMParameters()


INVALID_PARAMETERS: list[Callable[[], TTMParameters]] = [
    lambda: TTMParameters(context_length=64),
    lambda: TTMParameters(patch_stride=65),
    lambda: TTMParameters(model_size=190),
    lambda: TTMParameters(adaptive_patching_levels=-1),
    lambda: TTMParameters(dropout=1.0),
    lambda: TTMParameters(scale_epsilon=0.0),
    lambda: TTMParameters(decoder_num_layers=0),
]


@pytest.mark.parametrize("build", INVALID_PARAMETERS)
def test_invalid_parameters(build: Callable[[], TTMParameters]) -> None:
    with pytest.raises(ValueError):
        _ = build()
