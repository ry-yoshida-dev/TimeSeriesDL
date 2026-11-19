import importlib.util
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pathlib import Path

import huggingface_hub
import pytest
import torch
from torch import Tensor, nn

from time_series_dl.models.components.pretrained import (
    HubClient,
    MissingDependencyError,
    NameRule,
    ParameterNameMapping,
    PretrainedWeights,
    PretrainedWeightsLoader,
    SafetensorsFile,
    WeightsMismatchError,
)

WEIGHTS: PretrainedWeights = PretrainedWeights(repository="owner/model-name", revision="0123abcd")


class TwoLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first: nn.Linear = nn.Linear(3, 4)
        self.second: nn.Linear = nn.Linear(4, 2, bias=False)


MAPPING: ParameterNameMapping = ParameterNameMapping(
    rules=(NameRule(r"^layers\.0\.", "first."), NameRule(r"^layers\.1\.", "second.")),
    ignored=(r"head\..*",),
)


def published_tensors() -> dict[str, Tensor]:
    return {
        "layers.0.weight": torch.randn(4, 3),
        "layers.0.bias": torch.randn(4),
        "layers.1.weight": torch.randn(2, 4),
        "head.weight": torch.randn(7, 7),
    }


def test_weights_validate_repository_revision_and_file() -> None:
    assert WEIGHTS.filename == "model.safetensors"
    for repository in ("model", "owner/", "owner/name/extra", "owner name/model"):
        with pytest.raises(ValueError, match="repository"):
            _ = PretrainedWeights(repository=repository, revision="main")
    with pytest.raises(ValueError, match="revision"):
        _ = PretrainedWeights(repository="owner/model", revision="")
    for filename in ("model.bin", "/model.safetensors", "../model.safetensors"):
        with pytest.raises(ValueError, match="filename"):
            _ = PretrainedWeights(repository="owner/model", revision="main", filename=filename)


def test_name_rule_replaces_every_match_with_groups() -> None:
    rule = NameRule(r"block\.(\d+)\.", r"blocks.\1.")
    assert rule.apply("block.0.layer.block.12.weight") == "blocks.0.layer.blocks.12.weight"
    with pytest.raises(ValueError, match="regular expression"):
        _ = NameRule("(", "x")
    with pytest.raises(ValueError, match="empty"):
        _ = NameRule("", "x")


def test_mapping_renames_in_order_and_drops_ignored_names() -> None:
    mapping = ParameterNameMapping(rules=(NameRule("a", "b"), NameRule("b", "c")), ignored=(r"skip\..*",))
    assert mapping.rename("a.x") == "c.x"
    tensors: dict[str, Tensor] = {"a.x": torch.zeros(1), "skip.y": torch.zeros(1)}
    assert set(mapping.apply(tensors)) == {"c.x"}
    assert mapping.is_ignored("skip.y")
    assert not mapping.is_ignored("noskip.y")


def test_mapping_rejects_names_that_collide() -> None:
    mapping = ParameterNameMapping(rules=(NameRule(r"^(a|b)\.", "c."),))
    with pytest.raises(WeightsMismatchError, match="both map to c.x") as error:
        _ = mapping.apply({"a.x": torch.zeros(1), "b.x": torch.zeros(1)})
    assert error.value.unexpected


def test_loader_copies_renamed_tensors_and_converts_dtype() -> None:
    network = TwoLayers()
    tensors: dict[str, Tensor] = {name: tensor.double() for name, tensor in published_tensors().items()}
    PretrainedWeightsLoader.load_tensors(network, tensors, MAPPING)
    torch.testing.assert_close(network.first.weight, tensors["layers.0.weight"].float())
    torch.testing.assert_close(network.first.bias, tensors["layers.0.bias"].float())
    torch.testing.assert_close(network.second.weight, tensors["layers.1.weight"].float())
    assert network.first.weight.dtype == torch.float32


def drop_bias(tensors: dict[str, Tensor]) -> None:
    del tensors["layers.0.bias"]


def add_tensor(tensors: dict[str, Tensor]) -> None:
    tensors["layers.2.weight"] = torch.zeros(1)


def reshape_tensor(tensors: dict[str, Tensor]) -> None:
    tensors["layers.1.weight"] = torch.zeros(2, 5)


@pytest.mark.parametrize(
    ("change", "field"),
    [(drop_bias, "missing"), (add_tensor, "unexpected"), (reshape_tensor, "mismatched")],
)
def test_loader_rejects_inexact_weights(change: Callable[[dict[str, Tensor]], None], field: str) -> None:
    tensors: dict[str, Tensor] = published_tensors()
    change(tensors)
    network = TwoLayers()
    before: Tensor = network.first.weight.detach().clone()
    with pytest.raises(WeightsMismatchError) as error:
        PretrainedWeightsLoader.load_tensors(network, tensors, MAPPING)
    assert getattr(error.value, field)
    torch.testing.assert_close(network.first.weight, before)


def test_safetensors_file_round_trip(tmp_path: Path) -> None:
    path: Path = tmp_path / "model.safetensors"
    tensors: dict[str, Tensor] = published_tensors()
    tensors["transposed"] = torch.randn(3, 5).T
    SafetensorsFile.write(path, tensors)
    read: dict[str, Tensor] = SafetensorsFile.read(path)
    assert set(read) == set(tensors)
    for name, tensor in tensors.items():
        torch.testing.assert_close(read[name], tensor)
    network = TwoLayers()
    del tensors["transposed"]
    SafetensorsFile.write(path, tensors)
    PretrainedWeightsLoader.load_file(network, path, MAPPING)
    torch.testing.assert_close(network.second.weight, tensors["layers.1.weight"])
    with pytest.raises(FileNotFoundError):
        _ = SafetensorsFile.read(tmp_path / "absent.safetensors")


def test_hub_client_sends_only_a_fixed_user_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_download(**arguments: object) -> str:
        calls.append(arguments)
        return str(tmp_path / "model.safetensors")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert HubClient.download(WEIGHTS) == tmp_path / "model.safetensors"
    assert calls == [
        {
            "repo_id": "owner/model-name",
            "filename": "model.safetensors",
            "revision": "0123abcd",
            "headers": {"user-agent": "time-series-dl"},
        }
    ]


def test_missing_extra_raises_a_specific_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = importlib.util.find_spec

    def find_spec_without_extra(name: str, package: str | None = None) -> ModuleSpec | None:
        if name in ("huggingface_hub", "safetensors"):
            return None
        return original(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_extra)
    with pytest.raises(MissingDependencyError, match="pretrained"):
        _ = HubClient.download(WEIGHTS)
    path: Path = tmp_path / "model.safetensors"
    path.write_bytes(b"")
    with pytest.raises(MissingDependencyError, match="pretrained"):
        _ = SafetensorsFile.read(path)
    assert issubclass(MissingDependencyError, ImportError)
