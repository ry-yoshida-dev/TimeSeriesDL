from dataclasses import replace
from pathlib import Path

import pytest
import torch
from network_checks import check_forecast_network
from t5_published_names import published_stack_names
from torch import Tensor

from time_series_dl.models.architectures.foundation.chronos_bolt import (
    ChronosBoltForecastNetwork,
    ChronosBoltOfficialNames,
    ChronosBoltParameters,
    ChronosBoltSize,
)
from time_series_dl.models.components.pretrained import (
    HubClient,
    PretrainedWeights,
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)
from time_series_dl.models.components.t5 import T5FeedForwardKind
from time_series_dl.models.interfaces import ForecastShape
from time_series_dl.models.interfaces.protocols.zero_shot_forecasting import ZeroShotForecastingModel

TINY: ChronosBoltParameters = ChronosBoltParameters(
    model_size=8,
    key_value_size=4,
    num_heads=2,
    feedforward_size=12,
    num_encoder_layers=2,
    num_decoder_layers=1,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=16,
    context_length=32,
    patch_length=4,
    patch_stride=4,
    prediction_length=3,
    quantile_levels=(0.1, 0.5, 0.9),
)

QUANTILES: ForecastShape = ForecastShape(
    input_length=40,
    input_channels=3,
    horizon=7,
    target_channels=(2, 0),
    num_parameters=3,
    location_index=1,
    level_parameters=(0, 1, 2),
)


@pytest.mark.parametrize(
    "shape",
    [
        ForecastShape(input_length=13, input_channels=3, horizon=2, target_channels=(0, 1, 2)),
        QUANTILES,
        ForecastShape(input_length=4, input_channels=1, horizon=1, target_channels=(0,)),
    ],
)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(TINY, shape)


def build_network(shape: ForecastShape = QUANTILES) -> ChronosBoltForecastNetwork:
    _ = torch.manual_seed(0)
    network = TINY.build_forecast_network(shape)
    assert isinstance(network, ChronosBoltForecastNetwork)
    return network.eval()


def test_point_forecast_is_the_median() -> None:
    point_shape = ForecastShape(input_length=40, input_channels=3, horizon=7, target_channels=(2, 0))
    quantiles: ChronosBoltForecastNetwork = build_network()
    point: ChronosBoltForecastNetwork = build_network(point_shape)
    past: Tensor = torch.randn(2, 40, 3)
    with torch.no_grad():
        torch.testing.assert_close(point(past)[..., 0], quantiles(past)[..., 1])


def test_forecasts_are_affine_equivariant() -> None:
    network: ChronosBoltForecastNetwork = build_network()
    series: Tensor = torch.randn(4, 20)
    with torch.no_grad():
        forecast: Tensor = network.model(series)
        shifted: Tensor = network.model(3.0 * series + 100.0)
    torch.testing.assert_close(shifted, 3.0 * forecast + 100.0, rtol=1e-4, atol=1e-3)


def test_constant_series_use_the_epsilon_scale() -> None:
    network: ChronosBoltForecastNetwork = build_network()
    with torch.no_grad():
        forecast: Tensor = network.model(torch.full((1, 8), 5.0))
        zero: Tensor = network.model(torch.zeros(1, 8))
    torch.testing.assert_close(forecast, zero * 1.0 + 5.0)
    assert float((zero.abs()).max()) < 1e-3


def test_missing_patches_are_ignored() -> None:
    network: ChronosBoltForecastNetwork = build_network()
    observed: Tensor = torch.randn(2, 12)
    padded: Tensor = torch.cat([torch.full((2, 8), torch.nan), observed], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(network.model(padded), network.model(observed))


def test_long_horizons_append_every_quantile_path() -> None:
    network: ChronosBoltForecastNetwork = build_network()
    series: Tensor = torch.randn(2, 40)
    with torch.no_grad():
        forecast: Tensor = network.forecast(series)
        first: Tensor = network.model(series)
        paths: Tensor = torch.cat([series[:, -32:].unsqueeze(1).expand(-1, 3, -1), first], dim=-1)[..., -32:]
        samples: Tensor = network.model(paths.reshape(6, 32)).reshape(2, 9, 3)
        second: Tensor = torch.quantile(samples, torch.tensor([0.1, 0.5, 0.9]), dim=1).transpose(0, 1)
    assert forecast.shape == (2, 3, 7)
    torch.testing.assert_close(forecast[..., :3], first)
    torch.testing.assert_close(forecast[..., 3:6], second)


def test_rejects_unsupported_parameter_counts() -> None:
    shape = ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,), num_parameters=2)
    with pytest.raises(ValueError, match="num_parameters"):
        _ = TINY.build_forecast_network(shape)
    without_median = ChronosBoltParameters(quantile_levels=(0.1, 0.9))
    with pytest.raises(ValueError, match="0.5"):
        _ = without_median.build_forecast_network(
            ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
        )


def published_state_dict(parameters: ChronosBoltParameters) -> dict[str, Tensor]:
    """Random tensors under the names of chronos-forecasting's ``ChronosBoltModelForForecasting``."""
    model: int = parameters.model_size
    inner: int = parameters.num_heads * parameters.key_value_size
    hidden: int = parameters.feedforward_size
    patch: int = 2 * parameters.patch_length
    outputs: int = len(parameters.quantile_levels) * parameters.prediction_length
    shapes: dict[str, tuple[int, ...]] = {
        "shared.weight": (2, model),
        "input_patch_embedding.hidden_layer.weight": (hidden, patch),
        "input_patch_embedding.hidden_layer.bias": (hidden,),
        "input_patch_embedding.output_layer.weight": (model, hidden),
        "input_patch_embedding.output_layer.bias": (model,),
        "input_patch_embedding.residual_layer.weight": (model, patch),
        "input_patch_embedding.residual_layer.bias": (model,),
        "output_patch_embedding.hidden_layer.weight": (hidden, model),
        "output_patch_embedding.hidden_layer.bias": (hidden,),
        "output_patch_embedding.output_layer.weight": (outputs, hidden),
        "output_patch_embedding.output_layer.bias": (outputs,),
        "output_patch_embedding.residual_layer.weight": (outputs, model),
        "output_patch_embedding.residual_layer.bias": (outputs,),
    }
    stacks: list[str] = published_stack_names(
        "encoder.", parameters.num_encoder_layers, is_decoder=False, kind=T5FeedForwardKind.RELU
    ) + published_stack_names("decoder.", parameters.num_decoder_layers, is_decoder=True, kind=T5FeedForwardKind.RELU)
    for name in stacks:
        if name.endswith("relative_attention_bias.weight"):
            shapes[name] = (parameters.relative_attention_num_buckets, parameters.num_heads)
        elif name.endswith(".wi.weight"):
            shapes[name] = (hidden, model)
        elif name.endswith(".wo.weight"):
            shapes[name] = (model, hidden)
        elif name.endswith((".q.weight", ".k.weight", ".v.weight")):
            shapes[name] = (inner, model)
        elif name.endswith(".o.weight"):
            shapes[name] = (model, inner)
        else:
            shapes[name] = (model,)
    _ = torch.manual_seed(1)
    return {name: torch.randn(shape) for name, shape in shapes.items()}


def test_published_names_load_into_the_network(tmp_path: Path) -> None:
    path: Path = tmp_path / "model.safetensors"
    tensors: dict[str, Tensor] = published_state_dict(TINY)
    SafetensorsFile.write(path, tensors)
    network: ChronosBoltForecastNetwork = build_network()
    PretrainedWeightsLoader.load_file(network, path, ChronosBoltOfficialNames.mapping())
    model = network.model
    torch.testing.assert_close(model.token_embedding.weight, tensors["shared.weight"])
    torch.testing.assert_close(
        model.input_embedding.residual.weight, tensors["input_patch_embedding.residual_layer.weight"]
    )
    torch.testing.assert_close(model.output_projection.output.bias, tensors["output_patch_embedding.output_layer.bias"])
    torch.testing.assert_close(
        model.encoder.position_bias.embedding.weight,
        tensors["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
    )
    torch.testing.assert_close(
        model.encoder.state_dict()["blocks.1.feed_forward.output.weight"],
        tensors["encoder.block.1.layer.1.DenseReluDense.wo.weight"],
    )
    torch.testing.assert_close(
        model.decoder.state_dict()["blocks.0.cross_attention.key.weight"],
        tensors["decoder.block.0.layer.1.EncDecAttention.k.weight"],
    )
    torch.testing.assert_close(model.decoder.final_norm.weight, tensors["decoder.final_layer_norm.weight"])


def test_tied_embedding_copies_are_ignored_and_other_extras_rejected() -> None:
    tensors: dict[str, Tensor] = published_state_dict(TINY)
    tensors["encoder.embed_tokens.weight"] = tensors["shared.weight"]
    tensors["lm_head.weight"] = tensors["shared.weight"]
    PretrainedWeightsLoader.load_tensors(build_network(), tensors, ChronosBoltOfficialNames.mapping())
    tensors["encoder.block.2.layer.0.layer_norm.weight"] = torch.ones(8)
    with pytest.raises(WeightsMismatchError, match="unexpected"):
        PretrainedWeightsLoader.load_tensors(build_network(), tensors, ChronosBoltOfficialNames.mapping())
    del tensors["encoder.block.2.layer.0.layer_norm.weight"]
    tensors["shared.weight"] = torch.randn(1, 8)
    with pytest.raises(WeightsMismatchError, match="shape mismatch"):
        PretrainedWeightsLoader.load_tensors(build_network(), tensors, ChronosBoltOfficialNames.mapping())


def test_builder_loads_the_named_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path: Path = tmp_path / "model.safetensors"
    tensors: dict[str, Tensor] = published_state_dict(TINY)
    SafetensorsFile.write(path, tensors)
    requested: list[PretrainedWeights] = []

    def download(weights: PretrainedWeights) -> Path:
        requested.append(weights)
        return path

    monkeypatch.setattr(HubClient, "download", staticmethod(download))
    weights = PretrainedWeights(repository="amazon/chronos-bolt-tiny", revision="main")
    parameters = replace(TINY, weights=weights)
    network = parameters.build_forecast_network(QUANTILES)
    assert requested == [weights]
    assert isinstance(network, ChronosBoltForecastNetwork)
    torch.testing.assert_close(network.model.token_embedding.weight, tensors["shared.weight"])
    assert parameters.is_pretrained
    assert not parameters.without_pretrained_weights().is_pretrained
    _ = parameters.without_pretrained_weights().build_forecast_network(QUANTILES)
    assert len(requested) == 1


@pytest.mark.parametrize("size", list(ChronosBoltSize))
def test_pretrained_sizes_pin_their_repository(size: ChronosBoltSize) -> None:
    parameters = ChronosBoltParameters.pretrained(size)
    expected = PretrainedWeights(repository=f"amazon/chronos-bolt-{size.value}", revision=size.revision)
    assert parameters.weights == expected
    assert len(size.revision) == 40
    assert isinstance(parameters, ZeroShotForecastingModel)
    assert parameters.zero_shot_quantile_levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    small = ChronosBoltParameters.pretrained(ChronosBoltSize.SMALL)
    assert small.without_pretrained_weights() == ChronosBoltParameters()


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="patch_stride"):
        _ = ChronosBoltParameters(patch_length=8, patch_stride=16)
    with pytest.raises(ValueError, match="increasing"):
        _ = ChronosBoltParameters(quantile_levels=(0.5, 0.1))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        _ = ChronosBoltParameters(quantile_levels=(0.0, 0.5))
    with pytest.raises(ValueError, match="context_length"):
        _ = ChronosBoltParameters(context_length=0)
    with pytest.raises(ValueError, match="dropout"):
        _ = ChronosBoltParameters(dropout=1.0)
