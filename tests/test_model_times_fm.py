from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models.architectures.foundation.times_fm import (
    TimesFMDecoderLayer,
    TimesFMForecastNetwork,
    TimesFMFrequency,
    TimesFMModel,
    TimesFMOfficialNames,
    TimesFMParameters,
    TimesFMRelease,
)
from time_series_dl.models.components.pretrained import PretrainedWeightsLoader, SafetensorsFile, WeightsMismatchError
from time_series_dl.models.interfaces import ForecastShape
from time_series_dl.models.interfaces.protocols.zero_shot_forecasting import ZeroShotForecastingModel

TINY: TimesFMParameters = TimesFMParameters(
    model_size=8,
    num_heads=2,
    head_size=4,
    feedforward_size=16,
    num_layers=2,
    context_length=16,
    patch_length=4,
    output_patch_length=4,
    quantile_levels=(0.25, 0.5, 0.75),
)

FORECAST_SHAPES: list[ForecastShape] = [
    ForecastShape(input_length=12, input_channels=3, horizon=4, target_channels=(0, 1, 2)),
    ForecastShape(
        input_length=17,
        input_channels=4,
        horizon=10,
        target_channels=(2, 0),
        num_parameters=3,
        location_index=1,
        level_parameters=(0, 1, 2),
    ),
    ForecastShape(input_length=3, input_channels=1, horizon=1, target_channels=(0,)),
]


def official_state_dict(parameters: TimesFMParameters) -> dict[str, Tensor]:
    """Random tensors under the names of the transformers ``TimesFmModelForPrediction``."""
    model: int = parameters.model_size
    inner: int = parameters.num_heads * parameters.head_size
    hidden: int = parameters.feedforward_size
    outputs: int = parameters.output_patch_length * (1 + len(parameters.quantile_levels))
    shapes: dict[str, tuple[int, ...]] = {
        "decoder.freq_emb.weight": (parameters.num_frequencies, model),
        "decoder.input_ff_layer.input_layer.weight": (hidden, 2 * parameters.patch_length),
        "decoder.input_ff_layer.input_layer.bias": (hidden,),
        "decoder.input_ff_layer.output_layer.weight": (model, hidden),
        "decoder.input_ff_layer.output_layer.bias": (model,),
        "decoder.input_ff_layer.residual_layer.weight": (model, 2 * parameters.patch_length),
        "decoder.input_ff_layer.residual_layer.bias": (model,),
        "horizon_ff_layer.input_layer.weight": (hidden, model),
        "horizon_ff_layer.input_layer.bias": (hidden,),
        "horizon_ff_layer.output_layer.weight": (outputs, hidden),
        "horizon_ff_layer.output_layer.bias": (outputs,),
        "horizon_ff_layer.residual_layer.weight": (outputs, model),
        "horizon_ff_layer.residual_layer.bias": (outputs,),
    }
    for layer in range(parameters.num_layers):
        prefix: str = f"decoder.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (model,)
        shapes[prefix + "self_attn.scaling"] = (parameters.head_size,)
        for projection in ("q_proj", "k_proj", "v_proj"):
            shapes[prefix + f"self_attn.{projection}.weight"] = (inner, model)
            shapes[prefix + f"self_attn.{projection}.bias"] = (inner,)
        shapes[prefix + "self_attn.o_proj.weight"] = (model, inner)
        shapes[prefix + "self_attn.o_proj.bias"] = (model,)
        shapes[prefix + "mlp.layer_norm.weight"] = (model,)
        shapes[prefix + "mlp.layer_norm.bias"] = (model,)
        shapes[prefix + "mlp.gate_proj.weight"] = (hidden, model)
        shapes[prefix + "mlp.gate_proj.bias"] = (hidden,)
        shapes[prefix + "mlp.down_proj.weight"] = (model, hidden)
        shapes[prefix + "mlp.down_proj.bias"] = (model,)
    generator: torch.Generator = torch.Generator().manual_seed(0)
    return {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}


def build_network(horizon: int, num_parameters: int = 1) -> TimesFMForecastNetwork:
    shape: ForecastShape = ForecastShape(
        input_length=10,
        input_channels=1,
        horizon=horizon,
        target_channels=(0,),
        num_parameters=num_parameters,
        location_index=num_parameters // 2,
        level_parameters=tuple(range(num_parameters)),
    )
    network = TINY.build_forecast_network(shape)
    assert isinstance(network, TimesFMForecastNetwork)
    return network.eval()


def decoder_layer(model: TimesFMModel, index: int) -> TimesFMDecoderLayer:
    layer: object = model.layers[index]
    assert isinstance(layer, TimesFMDecoderLayer)
    return layer


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(TINY, shape)


def test_rejects_other_numbers_of_parameters() -> None:
    with pytest.raises(ValueError, match="num_parameters"):
        build_network(horizon=4, num_parameters=2)


def test_published_names_load_exactly(tmp_path: Path) -> None:
    tensors: dict[str, Tensor] = official_state_dict(TINY)
    path: Path = tmp_path / "model.safetensors"
    SafetensorsFile.write(path, tensors)
    network: TimesFMForecastNetwork = build_network(horizon=4)
    PretrainedWeightsLoader.load_file(network, path, TimesFMOfficialNames.mapping())
    model: TimesFMModel = network.model
    first: TimesFMDecoderLayer = decoder_layer(model, 0)
    second: TimesFMDecoderLayer = decoder_layer(model, 1)
    pairs: list[tuple[str, Tensor]] = [
        ("decoder.freq_emb.weight", model.frequency_embedding.weight),
        ("decoder.input_ff_layer.residual_layer.bias", model.input_embedding.residual.bias),
        ("horizon_ff_layer.input_layer.weight", model.output_projection.hidden.weight),
        ("decoder.layers.1.self_attn.k_proj.weight", second.attention.key.weight),
        ("decoder.layers.1.self_attn.scaling", second.attention.query_scale),
        ("decoder.layers.0.input_layernorm.weight", first.attention_norm.weight),
        ("decoder.layers.0.mlp.layer_norm.bias", first.feed_forward.norm.bias),
        ("decoder.layers.0.mlp.down_proj.weight", first.feed_forward.output.weight),
    ]
    for name, parameter in pairs:
        assert torch.equal(parameter, tensors[name]), name


def test_published_names_must_fit(tmp_path: Path) -> None:
    complete: dict[str, Tensor] = official_state_dict(TINY)
    missing: dict[str, Tensor] = {name: tensor for name, tensor in complete.items() if "scaling" not in name}
    unexpected: dict[str, Tensor] = {**complete, "decoder.position_emb.inv_timescales": torch.zeros(4)}
    reshaped: dict[str, Tensor] = {**complete, "decoder.freq_emb.weight": torch.zeros(5, TINY.model_size)}
    for tensors, field in ((missing, "missing"), (unexpected, "unexpected"), (reshaped, "mismatched")):
        path: Path = tmp_path / f"{field}.safetensors"
        SafetensorsFile.write(path, tensors)
        with pytest.raises(WeightsMismatchError) as error:
            PretrainedWeightsLoader.load_file(build_network(horizon=4), path, TimesFMOfficialNames.mapping())
        assert getattr(error.value, field)


def test_forecasts_are_affine_equivariant() -> None:
    torch.manual_seed(0)
    network: TimesFMForecastNetwork = build_network(horizon=4, num_parameters=3)
    series: Tensor = torch.randn(2, 10)
    with torch.no_grad():
        forecast: Tensor = network.forecast(series)
        shifted: Tensor = network.forecast(3.0 * series + 7.0)
    torch.testing.assert_close(shifted, 3.0 * forecast + 7.0, rtol=1e-4, atol=1e-4)


def test_padding_to_whole_patches_equals_padding_to_the_context_length() -> None:
    torch.manual_seed(0)
    model: TimesFMModel = build_network(horizon=4).model
    series: Tensor = torch.randn(2, 10)
    padded: Tensor = torch.cat([torch.zeros(2, 6), series], dim=1)
    is_padded: Tensor = torch.cat([torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 10, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        short: Tensor = model(series, torch.zeros(2, 10, dtype=torch.bool))
        long: Tensor = model(padded, is_padded)
    torch.testing.assert_close(short, long)


def test_statistics_come_from_the_first_patch_with_three_observed_values() -> None:
    torch.manual_seed(0)
    model: TimesFMModel = build_network(horizon=4).model
    series: Tensor = torch.randn(1, 12)
    # The first patch has two observed values only, so the second patch sets location and scale; rescaling the whole
    # series by the second patch's statistics therefore leaves the normalized inputs unchanged.
    is_padded: Tensor = torch.tensor([[True, True, False, False] + [False] * 8])
    second: Tensor = series[0, 4:8]
    location: Tensor = second.mean()
    scale: Tensor = second.std(unbiased=False)
    with torch.no_grad():
        original: Tensor = model(series, is_padded)
        standardized: Tensor = model((series - location) / scale, is_padded)
    torch.testing.assert_close(original, standardized * scale + location, rtol=1e-4, atol=1e-4)


def test_long_horizons_append_the_mean_forecast() -> None:
    torch.manual_seed(0)
    network: TimesFMForecastNetwork = build_network(horizon=7, num_parameters=3)
    series: Tensor = torch.randn(2, 14)
    no_padding: Tensor = torch.zeros(2, 16, dtype=torch.bool)
    with torch.no_grad():
        forecast: Tensor = network.forecast(series)
        first: Tensor = network.model(series, no_padding[:, :14])
        extended: Tensor = torch.cat([series, first[..., 0]], dim=1)[:, -16:]
        second: Tensor = network.model(extended, no_padding)
    torch.testing.assert_close(forecast, torch.cat([first, second], dim=1)[:, :7])


def test_point_forecast_is_the_mean_head_and_quantiles_follow() -> None:
    torch.manual_seed(0)
    point: TimesFMForecastNetwork = build_network(horizon=4)
    quantiles: TimesFMForecastNetwork = build_network(horizon=4, num_parameters=3)
    _ = quantiles.load_state_dict(point.state_dict())
    past: Tensor = torch.randn(2, 10, 1)
    with torch.no_grad():
        full: Tensor = point.forecast(past[..., 0])
        assert torch.equal(point(past)[:, :, 0, 0], full[..., 0])
        assert torch.equal(quantiles(past)[:, :, 0, :], full[..., 1:])


def test_pretrained_release() -> None:
    parameters: TimesFMParameters = TimesFMParameters.pretrained(TimesFMRelease.V2_0_500M, TimesFMFrequency.LOW)
    assert isinstance(parameters, ZeroShotForecastingModel)
    assert parameters.is_pretrained
    assert parameters.weights is not None
    assert parameters.weights.repository == "google/timesfm-2.0-500m-pytorch"
    assert parameters.weights.revision == "dc2443792ce5516872b89b37cf1bc058c3bf0c10"
    assert parameters.frequency == TimesFMFrequency.LOW
    assert parameters.without_pretrained_weights() == TimesFMParameters(frequency=TimesFMFrequency.LOW)
    assert parameters.zero_shot_quantile_levels == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_invalid_parameters_raise() -> None:
    invalid: list[Callable[[], TimesFMParameters]] = [
        lambda: TimesFMParameters(num_layers=0),
        lambda: TimesFMParameters(context_length=2, patch_length=4),
        lambda: TimesFMParameters(quantile_levels=()),
        lambda: TimesFMParameters(quantile_levels=(0.5, 0.5)),
        lambda: TimesFMParameters(quantile_levels=(0.0, 0.5)),
        lambda: TimesFMParameters(num_frequencies=2, frequency=TimesFMFrequency.LOW),
        lambda: TimesFMParameters(layer_norm_epsilon=0.0),
    ]
    for build in invalid:
        with pytest.raises(ValueError):
            _ = build()
