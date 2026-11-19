import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from masked_network_checks import check_masked_reconstruction_network, check_reads_observed_mask
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models.architectures.foundation.moment import (
    MOMENTEncoder,
    MOMENTEncoding,
    MOMENTForecastNetwork,
    MOMENTOfficialNames,
    MOMENTParameters,
    MOMENTReconstructionNetwork,
    MOMENTSize,
    MOMENTSummaryNetwork,
)
from time_series_dl.models.components.pretrained import (
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)
from time_series_dl.models.components.t5 import T5EncoderBlock
from time_series_dl.models.interfaces import ForecastShape, SequenceShape, SummaryShape

TINY: MOMENTParameters = MOMENTParameters(
    model_size=16,
    key_value_size=4,
    num_heads=2,
    feedforward_size=24,
    num_layers=2,
    input_length=32,
    patch_length=4,
    is_backbone_frozen=False,
)
SEQUENCE: SequenceShape = SequenceShape(length=30, channels=2)


def official_state_dict(parameters: MOMENTParameters) -> dict[str, Tensor]:
    """Random weights named as in the published MOMENT checkpoints (momentfm with a T5 encoder)."""
    generator: torch.Generator = torch.Generator().manual_seed(1)
    size: int = parameters.model_size
    inner: int = parameters.num_heads * parameters.key_value_size
    shapes: dict[str, tuple[int, ...]] = {
        "patch_embedding.value_embedding.weight": (size, parameters.patch_length),
        "patch_embedding.mask_embedding": (size,),
        "patch_embedding.position_embedding.pe": (1, 5000, size),
        "encoder.embed_tokens.weight": (100, size),
        "encoder.final_layer_norm.weight": (size,),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (
            parameters.relative_attention_num_buckets,
            parameters.num_heads,
        ),
        "head.linear.weight": (parameters.patch_length, size),
        "head.linear.bias": (parameters.patch_length,),
    }
    for block in range(parameters.num_layers):
        prefix: str = f"encoder.block.{block}.layer."
        for projection in ("q", "k", "v"):
            shapes[f"{prefix}0.SelfAttention.{projection}.weight"] = (inner, size)
        shapes[f"{prefix}0.SelfAttention.o.weight"] = (size, inner)
        shapes[f"{prefix}0.layer_norm.weight"] = (size,)
        shapes[f"{prefix}1.DenseReluDense.wi_0.weight"] = (parameters.feedforward_size, size)
        shapes[f"{prefix}1.DenseReluDense.wi_1.weight"] = (parameters.feedforward_size, size)
        shapes[f"{prefix}1.DenseReluDense.wo.weight"] = (size, parameters.feedforward_size)
        shapes[f"{prefix}1.layer_norm.weight"] = (size,)
    return {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(TINY, shape)


def test_forecast_network_reads_the_last_input_length_steps() -> None:
    shape = ForecastShape(input_length=50, input_channels=2, horizon=3, target_channels=(1,))
    network = TINY.build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 50, 2)
    changed: Tensor = past.clone()
    changed[:, :18] = 100.0
    with torch.no_grad():
        assert torch.equal(network(past), network(changed))


def test_summary_network() -> None:
    check_summary_network(TINY, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_reconstruction_network() -> None:
    check_reconstruction_network(TINY, SEQUENCE)


def test_masked_reconstruction_network() -> None:
    check_masked_reconstruction_network(TINY, SEQUENCE)
    check_reads_observed_mask(TINY, SEQUENCE)


def test_sequences_longer_than_the_input_are_rejected() -> None:
    with pytest.raises(ValueError, match="at most 32"):
        TINY.build_reconstruction_network(SequenceShape(length=33, channels=1))
    with pytest.raises(ValueError, match="at most 32"):
        TINY.build_summary_network(SummaryShape(input_length=40, input_channels=1, output_size=2))


def test_partly_padded_patch_is_hidden_and_excluded_from_the_statistics() -> None:
    _ = torch.manual_seed(0)
    network = TINY.build_reconstruction_network(SEQUENCE).eval()
    values: Tensor = torch.randn(2, 30, 2)
    changed: Tensor = values.clone()
    # 30 steps are padded by 2, so the first patch holds 2 padding and 2 real values.
    changed[:, :2] = 50.0
    with torch.no_grad():
        assert torch.allclose(network(values), network(changed))


def test_missing_values_are_hidden() -> None:
    _ = torch.manual_seed(0)
    network = TINY.build_masked_reconstruction_network(SEQUENCE).eval()
    values: Tensor = torch.randn(1, 30, 2)
    is_observed: Tensor = torch.ones(1, 30, 2, dtype=torch.bool)
    is_observed[0, 10, 1] = False
    changed: Tensor = values.clone()
    changed[0, 10, 1] = -40.0
    with torch.no_grad():
        assert torch.allclose(network(values, is_observed).values, network(changed, is_observed).values)


def test_normalization_uses_the_observed_values() -> None:
    encoder = MOMENTEncoder(TINY).eval()
    values: Tensor = torch.randn(2, 32) * 3.0 + 5.0
    is_observed: Tensor = torch.rand(2, 32) < 0.6
    is_observed[:, 0] = True
    is_input: Tensor = torch.ones(2, 32, dtype=torch.bool)
    with torch.no_grad():
        encoding: MOMENTEncoding = encoder(values, is_observed, is_observed, is_input)
    for row in range(2):
        observed: np.ndarray = values[row][is_observed[row]].numpy().astype(np.float64)
        assert encoding.location[row, 0].item() == pytest.approx(observed.mean(), rel=1e-5)
        assert encoding.scale[row, 0].item() == pytest.approx(observed.std() + 1e-5, rel=1e-5)
    restored: Tensor = encoding.denormalize((values - encoding.location) / encoding.scale)
    assert torch.allclose(restored, values, atol=1e-5)


def test_position_table_is_sinusoidal() -> None:
    table: Tensor = MOMENTEncoder(TINY).patch_embedding.position_table
    assert table.shape == (8, 16)
    assert table[3, 4].item() == pytest.approx(math.sin(3 * math.exp(-4 * math.log(10000.0) / 16)), abs=1e-6)
    assert table[3, 5].item() == pytest.approx(math.cos(3 * math.exp(-4 * math.log(10000.0) / 16)), abs=1e-6)


def test_frozen_backbone_trains_only_the_head() -> None:
    network = replace(TINY, is_backbone_frozen=True).build_reconstruction_network(SEQUENCE)
    assert isinstance(network, MOMENTReconstructionNetwork)
    assert not any(parameter.requires_grad for parameter in network.encoder.parameters())
    assert all(parameter.requires_grad for parameter in network.head.parameters())


def test_official_names_load_into_the_reconstruction_network(tmp_path: Path) -> None:
    official: dict[str, Tensor] = official_state_dict(TINY)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, official)
    network = TINY.build_reconstruction_network(SEQUENCE)
    assert isinstance(network, MOMENTReconstructionNetwork)
    PretrainedWeightsLoader.load_file(network, path, MOMENTOfficialNames.reconstruction_mapping())
    transformer = network.encoder.transformer
    blocks: list[T5EncoderBlock] = [cast(T5EncoderBlock, block) for block in transformer.blocks]
    gate: nn.Linear | None = blocks[0].feed_forward.gate
    assert gate is not None
    pairs: list[tuple[str, Tensor]] = [
        ("patch_embedding.value_embedding.weight", network.encoder.patch_embedding.value_embedding.weight),
        ("patch_embedding.mask_embedding", network.encoder.patch_embedding.mask_embedding),
        (
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            transformer.position_bias.embedding.weight,
        ),
        ("encoder.block.1.layer.0.SelfAttention.q.weight", blocks[1].self_attention.query.weight),
        ("encoder.block.1.layer.0.SelfAttention.o.weight", blocks[1].self_attention.output.weight),
        ("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", gate.weight),
        ("encoder.block.0.layer.1.DenseReluDense.wi_1.weight", blocks[0].feed_forward.input.weight),
        ("encoder.block.0.layer.1.layer_norm.weight", blocks[0].feed_forward_norm.weight),
        ("encoder.final_layer_norm.weight", transformer.final_norm.weight),
        ("head.linear.weight", network.head.linear.weight),
        ("head.linear.bias", network.head.linear.bias),
    ]
    for name, parameter in pairs:
        assert torch.equal(parameter.detach(), official[name]), name


def test_official_names_load_into_the_encoder_of_new_heads(tmp_path: Path) -> None:
    official: dict[str, Tensor] = official_state_dict(TINY)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, official)
    shape = ForecastShape(input_length=32, input_channels=1, horizon=4, target_channels=(0,))
    forecast = TINY.build_forecast_network(shape)
    assert isinstance(forecast, MOMENTForecastNetwork)
    PretrainedWeightsLoader.load_file(forecast.encoder, path, MOMENTOfficialNames.encoder_mapping())
    assert torch.equal(
        forecast.encoder.patch_embedding.value_embedding.weight.detach(),
        official["patch_embedding.value_embedding.weight"],
    )
    summary = TINY.build_summary_network(SummaryShape(input_length=32, input_channels=1, output_size=2))
    assert isinstance(summary, MOMENTSummaryNetwork)
    PretrainedWeightsLoader.load_file(summary.encoder, path, MOMENTOfficialNames.encoder_mapping())
    with pytest.raises(WeightsMismatchError, match=r"unexpected: .*head\.linear\.weight"):
        PretrainedWeightsLoader.load_file(forecast.encoder, path, MOMENTOfficialNames.reconstruction_mapping())


def test_mismatched_official_weights_raise() -> None:
    network = TINY.build_reconstruction_network(SEQUENCE)
    mapping = MOMENTOfficialNames.reconstruction_mapping()
    official: dict[str, Tensor] = official_state_dict(TINY)
    missing: dict[str, Tensor] = {name: tensor for name, tensor in official.items() if name != "head.linear.bias"}
    with pytest.raises(WeightsMismatchError, match="missing: head.linear.bias"):
        PretrainedWeightsLoader.load_tensors(network, missing, mapping)
    with pytest.raises(WeightsMismatchError, match="unexpected: encoder.extra"):
        PretrainedWeightsLoader.load_tensors(network, {**official, "encoder.extra": torch.zeros(1)}, mapping)
    resized: dict[str, Tensor] = {**official, "patch_embedding.mask_embedding": torch.zeros(17)}
    with pytest.raises(WeightsMismatchError, match="shape mismatch"):
        PretrainedWeightsLoader.load_tensors(network, resized, mapping)


def test_pretrained_names_published_checkpoints() -> None:
    small = MOMENTParameters.pretrained(MOMENTSize.SMALL)
    assert small.is_pretrained
    assert small.weights is not None and small.weights.repository == "AutonLab/MOMENT-1-small"
    assert (small.model_size, small.num_heads, small.feedforward_size, small.num_layers) == (512, 6, 1024, 8)
    assert small.is_backbone_frozen
    large = MOMENTParameters.pretrained(MOMENTSize.LARGE).without_pretrained_weights()
    assert replace(large, is_backbone_frozen=False) == MOMENTParameters()
    assert not small.without_pretrained_weights().is_pretrained


@pytest.mark.parametrize(
    "build",
    [
        lambda: MOMENTParameters(patch_length=7),
        lambda: MOMENTParameters(input_length=0),
        lambda: MOMENTParameters(head_dropout=1.0),
        lambda: MOMENTParameters(normalization_epsilon=0.0),
        lambda: MOMENTParameters(num_heads=0),
    ],
)
def test_invalid_parameters_raise(build: Callable[[], MOMENTParameters]) -> None:
    with pytest.raises(ValueError):
        _ = build()
