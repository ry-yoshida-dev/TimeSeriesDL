from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import yaml

import time_series_dl
from time_series_dl.models import (
    Activation,
    EncoderHead,
    MiniRocketParameters,
    ModelArchitecture,
    ModelParameters,
    ModelSettings,
    NBEATSBlockKind,
    NBEATSParameters,
    NBEATSStack,
    PatchTSTParameters,
    PretrainedEncoderParameters,
    SAITSParameters,
    TCNParameters,
    TemporalPooling,
    TS2VecParameters,
    USADParameters,
)
from time_series_dl.models.settings import ParametersReader

PRESET_DIRECTORY: Path = Path(time_series_dl.__file__).parent / "config"
PRESET_PATHS: list[Path] = sorted(PRESET_DIRECTORY.glob("*.yaml"))


def load_model_section(path: Path) -> dict[str, object]:
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    # YAML mappings are untyped, so give the keys and values the widest type.
    section: object = cast(dict[object, object], document)["model"]
    assert isinstance(section, dict)
    return {str(key): value for key, value in cast(dict[object, object], section).items()}


def default_parameters(architecture: ModelArchitecture) -> ModelParameters:
    # A pretrained encoder has no default file; its preset names the conventional one.
    if architecture is ModelArchitecture.PRETRAINED_ENCODER:
        return PretrainedEncoderParameters(checkpoint_path="representation_learner.pt")
    return ParametersReader.read(architecture.parameters_type, {})


def test_every_architecture_has_a_preset() -> None:
    assert {path.stem for path in PRESET_PATHS} == {architecture.value for architecture in ModelArchitecture}


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda path: path.stem)
def test_preset_builds_default_parameters(path: Path) -> None:
    section: dict[str, object] = load_model_section(path)
    assert set(section) == {field.name for field in fields(ModelSettings)}
    architecture: object = section["architecture"]
    parameters: object = section["parameters"]
    assert isinstance(architecture, str)
    assert isinstance(parameters, dict)
    settings = ModelSettings(
        architecture=ModelArchitecture(architecture),
        parameters=cast(dict[str, object], parameters),
    )
    assert settings.architecture.value == path.stem
    assert settings.build() == default_parameters(settings.architecture)


def test_nested_values_are_converted() -> None:
    settings = ModelSettings(
        architecture=ModelArchitecture.N_BEATS,
        parameters={"stacks": [{"kind": "trend", "num_blocks": 3}, {"kind": "seasonality"}]},
    )
    assert settings.build() == NBEATSParameters(
        stacks=(NBEATSStack(kind=NBEATSBlockKind.TREND, num_blocks=3), NBEATSStack(kind=NBEATSBlockKind.SEASONALITY))
    )
    tcn = ModelSettings(
        architecture=ModelArchitecture.TCN,
        parameters={"channel_sizes": [8, 8], "dropout": 0, "head": {"pooling": "mean"}},
    ).build()
    assert tcn == TCNParameters(channel_sizes=(8, 8), dropout=0.0, head=EncoderHead(pooling=TemporalPooling.MEAN))


def test_missing_parameters_keep_defaults() -> None:
    settings = ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"activation": "relu"})
    assert settings.build() == PatchTSTParameters(activation=Activation.RELU)
    assert ModelSettings(architecture=ModelArchitecture.PATCH_TST).build() == PatchTSTParameters()


def test_task_models_follow_supported_tasks() -> None:
    patch_tst = ModelSettings(architecture=ModelArchitecture.PATCH_TST)
    assert isinstance(patch_tst.build_summary_model(), PatchTSTParameters)
    assert isinstance(patch_tst.build_reconstruction_model(), PatchTSTParameters)
    n_beats = ModelSettings(architecture=ModelArchitecture.N_BEATS)
    assert isinstance(n_beats.build_forecasting_model(), NBEATSParameters)
    with pytest.raises(ValueError, match="classification"):
        n_beats.build_summary_model()
    with pytest.raises(ValueError, match="imputation"):
        n_beats.build_reconstruction_model()
    mini_rocket = ModelSettings(architecture=ModelArchitecture.MINI_ROCKET)
    assert isinstance(mini_rocket.build_summary_model(), MiniRocketParameters)
    with pytest.raises(ValueError, match="forecasting"):
        mini_rocket.build_forecasting_model()
    saits = ModelSettings(architecture=ModelArchitecture.SAITS)
    assert isinstance(saits.build_imputation_model(), SAITSParameters)
    with pytest.raises(ValueError, match="anomaly detection"):
        saits.build_reconstruction_model()
    with pytest.raises(ValueError, match="imputation"):
        mini_rocket.build_imputation_model()
    assert isinstance(ModelSettings(architecture=ModelArchitecture.TS2VEC).build_pretraining_model(), TS2VecParameters)
    with pytest.raises(ValueError, match="pretraining"):
        mini_rocket.build_pretraining_model()
    usad = ModelSettings(architecture=ModelArchitecture.USAD).build_anomaly_detection_model()
    assert isinstance(usad, USADParameters)
    assert isinstance(n_beats.build_anomaly_detection_model(), NBEATSParameters)
    with pytest.raises(ValueError, match="anomaly detection"):
        mini_rocket.build_anomaly_detection_model()
    assert isinstance(patch_tst.build_imputation_model(), PatchTSTParameters)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(KeyError, match="patch_size"):
        ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"patch_size": 16})
    with pytest.raises(TypeError, match="patch_length"):
        ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"patch_length": "16"})
    with pytest.raises(TypeError, match="patch_length"):
        ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"patch_length": True})
    with pytest.raises(ValueError, match="swish"):
        ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"activation": "swish"})
    with pytest.raises(ValueError, match="num_heads"):
        ModelSettings(architecture=ModelArchitecture.PATCH_TST, parameters={"num_heads": 3})
    with pytest.raises(TypeError, match="channel_sizes"):
        ModelSettings(architecture=ModelArchitecture.TCN, parameters={"channel_sizes": "64"})
