from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from .hub_client import HubClient
from .name_mapping import ParameterNameMapping
from .safetensors_file import SafetensorsFile
from .weights import PretrainedWeights
from .weights_mismatch_error import WeightsMismatchError


class PretrainedWeightsLoader:
    """
    Loads published weights into a network, requiring an exact fit.

    After renaming, every parameter and persistent buffer of the network must receive exactly one tensor of its
    shape, and every published tensor that is not explicitly ignored must find its parameter; tensors are converted
    to the dtype and device of the parameter they replace.
    """

    @staticmethod
    def load(network: nn.Module, weights: PretrainedWeights, mapping: ParameterNameMapping) -> None:
        """
        Download (or read from the cache) published weights and load them.

        Parameters
        ----------
        network : nn.Module
            The network, built with the architecture of the published model.
        weights : PretrainedWeights
            The repository, revision and file.
        mapping : ParameterNameMapping
            Translation of the published names into the network's names.

        Raises
        ------
        MissingDependencyError
            If the optional ``pretrained`` dependencies are not installed.
        WeightsMismatchError
            If the weights do not fit the network exactly.
        """
        PretrainedWeightsLoader.load_file(network, HubClient.download(weights), mapping)

    @staticmethod
    def load_file(network: nn.Module, path: Path, mapping: ParameterNameMapping) -> None:
        """
        Load published weights from a local safetensors file.

        Parameters
        ----------
        network : nn.Module
            The network, built with the architecture of the published model.
        path : Path
            The safetensors file, with the published names.
        mapping : ParameterNameMapping
            Translation of the published names into the network's names.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        MissingDependencyError
            If ``safetensors`` is not installed.
        WeightsMismatchError
            If the weights do not fit the network exactly.
        """
        PretrainedWeightsLoader.load_tensors(network, SafetensorsFile.read(path), mapping)

    @staticmethod
    def load_tensors(network: nn.Module, tensors: Mapping[str, Tensor], mapping: ParameterNameMapping) -> None:
        """
        Load published weights that are already in memory.

        Parameters
        ----------
        network : nn.Module
            The network, built with the architecture of the published model.
        tensors : Mapping[str, Tensor]
            The weights by published name.
        mapping : ParameterNameMapping
            Translation of the published names into the network's names.

        Raises
        ------
        WeightsMismatchError
            If two published names translate to the same name, or the weights do not fit the network exactly.
        """
        renamed: dict[str, Tensor] = mapping.apply(tensors)
        expected: dict[str, Tensor] = network.state_dict()
        missing: list[str] = sorted(set(expected) - set(renamed))
        unexpected: list[str] = sorted(set(renamed) - set(expected))
        mismatched: list[str] = [
            f"{name} {tuple(renamed[name].shape)} != {tuple(expected[name].shape)}"
            for name in sorted(set(renamed) & set(expected))
            if renamed[name].shape != expected[name].shape
        ]
        if missing or unexpected or mismatched:
            raise WeightsMismatchError(missing=missing, unexpected=unexpected, mismatched=mismatched)
        converted: dict[str, Tensor] = {
            name: tensor.to(device=expected[name].device, dtype=expected[name].dtype)
            for name, tensor in renamed.items()
        }
        with torch.no_grad():
            _ = network.load_state_dict(converted, strict=True)
