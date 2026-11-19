from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from data_scaler import ScalerState, ScalingStrategy
from numpy.typing import NDArray
from torch import Tensor, nn

from .encoded_value import EncodedValue
from .task_kind import TaskKind

FORMAT_VERSION: Final[int] = 1


@dataclass(frozen=True, eq=False)
class Checkpoint:
    """
    Everything a fitted task needs to predict again, written to one file that loads without pickle.

    The file is read by ``torch.load`` with ``weights_only=True``, so it holds only tensors and plain values, and
    loading an untrusted file cannot run code.

    Attributes
    ----------
    task : TaskKind
        The task the checkpoint was saved from.
    settings : EncodedValue
        The settings of the task, as written by ``SettingsCodec``.
    history : EncodedValue
        The ``TrainingHistory`` of the fit, as written by ``SettingsCodec``.
    values : Mapping[str, EncodedValue]
        What the task learned from the training data other than weights and scaling, such as the number of channels.
    scalers : Mapping[str, ScalerState]
        The fitted scalers by name.
    network : Mapping[str, Tensor]
        The state dict of the trained network.
    """

    task: TaskKind
    settings: EncodedValue
    history: EncodedValue
    values: Mapping[str, EncodedValue]
    scalers: Mapping[str, ScalerState]
    network: Mapping[str, Tensor]

    def save(self, path: Path) -> None:
        """
        Write the checkpoint to a file.

        Parameters
        ----------
        path : Path
            The file to write; ``load`` reads it back.
        """
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "task": self.task.value,
                "settings": self.settings,
                "history": self.history,
                "values": dict(self.values),
                "scalers": {name: self._encode_scaler(state) for name, state in self.scalers.items()},
                "network": {name: tensor.detach().cpu() for name, tensor in self.network.items()},
            },
            path,
        )

    @staticmethod
    def load(path: Path, task: TaskKind) -> Checkpoint:
        """
        Read a checkpoint written by save.

        Parameters
        ----------
        path : Path
            The file to read.
        task : TaskKind
            The task the checkpoint must have been saved from.

        Returns
        -------
        Checkpoint
            The checkpoint, with every tensor on the CPU.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a checkpoint of a supported format version or was saved from another task.
        """
        location: str = str(path)
        loaded: object = cast(object, torch.load(path, map_location="cpu", weights_only=True))
        root: dict[str, object] = Checkpoint._require_dictionary(loaded, location)
        version: object = root.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"{location}: unsupported checkpoint format version {version!r}")
        saved_task: object = root.get("task")
        if saved_task != task.value:
            raise ValueError(f"{location} was saved from the {saved_task!r} task, not {task.value!r}")
        values: dict[str, object] = Checkpoint._require_dictionary(root.get("values"), f"{location}: values")
        scalers: dict[str, object] = Checkpoint._require_dictionary(root.get("scalers"), f"{location}: scalers")
        network: dict[str, object] = Checkpoint._require_dictionary(root.get("network"), f"{location}: network")
        return Checkpoint(
            task=task,
            settings=Checkpoint._require_encoded(root.get("settings"), f"{location}: settings"),
            history=Checkpoint._require_encoded(root.get("history"), f"{location}: history"),
            values={
                name: Checkpoint._require_encoded(value, f"{location}: values.{name}") for name, value in values.items()
            },
            scalers={
                name: Checkpoint._decode_scaler(state, f"{location}: scalers.{name}") for name, state in scalers.items()
            },
            network={
                name: Checkpoint._require_tensor(tensor, f"{location}: network.{name}")
                for name, tensor in network.items()
            },
        )

    def restore_network(self, network: nn.Module, device: torch.device) -> None:
        """
        Load the saved weights into a network built with the saved shape, and prepare it for inference.

        Parameters
        ----------
        network : nn.Module
            The untrained network, built like the saved one.
        device : torch.device
            The device the network is moved to.

        Raises
        ------
        ValueError
            If the saved weights do not match the parameters and buffers of the network.
        """
        expected: set[str] = set(network.state_dict())
        if set(self.network) != expected:
            missing: list[str] = sorted(expected - set(self.network))
            unexpected: list[str] = sorted(set(self.network) - expected)
            raise ValueError(f"checkpoint weights do not match the network: missing {missing}, unexpected {unexpected}")
        _ = network.load_state_dict(self.network)
        _ = network.to(device)
        _ = network.eval()

    def read_int(self, name: str) -> int:
        """
        Get an integer of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        int
            The value.

        Raises
        ------
        ValueError
            If the value is missing or not an integer.
        """
        value: EncodedValue = self._read(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"checkpoint value {name!r} must be an integer, got {value!r}")
        return value

    def read_float(self, name: str) -> float:
        """
        Get a real number of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        float
            The value.

        Raises
        ------
        ValueError
            If the value is missing or not a real number.
        """
        value: EncodedValue = self._read(name)
        if not isinstance(value, float):
            raise ValueError(f"checkpoint value {name!r} must be a real number, got {value!r}")
        return value

    def read_bool(self, name: str) -> bool:
        """
        Get a boolean of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        bool
            The value.

        Raises
        ------
        ValueError
            If the value is missing or not a boolean.
        """
        value: EncodedValue = self._read(name)
        if not isinstance(value, bool):
            raise ValueError(f"checkpoint value {name!r} must be a boolean, got {value!r}")
        return value

    def read_str(self, name: str) -> str:
        """
        Get a string of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        str
            The value.

        Raises
        ------
        ValueError
            If the value is missing or not a string.
        """
        value: EncodedValue = self._read(name)
        if not isinstance(value, str):
            raise ValueError(f"checkpoint value {name!r} must be a string, got {value!r}")
        return value

    def read_list(self, name: str) -> list[EncodedValue]:
        """
        Get a list of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        list[EncodedValue]
            The value.

        Raises
        ------
        ValueError
            If the value is missing or not a list.
        """
        value: EncodedValue = self._read(name)
        if not isinstance(value, list):
            raise ValueError(f"checkpoint value {name!r} must be a list, got {value!r}")
        return value

    def read_int_tuple(self, name: str) -> tuple[int, ...]:
        """
        Get a list of integers of ``values``.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        tuple[int, ...]
            The integers.

        Raises
        ------
        ValueError
            If the value is missing or not a list of integers.
        """
        items: list[EncodedValue] = self.read_list(name)
        integers: list[int] = [item for item in items if isinstance(item, int) and not isinstance(item, bool)]
        if len(integers) != len(items):
            raise ValueError(f"checkpoint value {name!r} must be a list of integers, got {items!r}")
        return tuple(integers)

    def contains(self, name: str) -> bool:
        """
        Check whether ``values`` has a value, for values that only some checkpoints hold.

        Parameters
        ----------
        name : str
            The name of the value.

        Returns
        -------
        bool
            True if the value is present.
        """
        return name in self.values

    def read_scaler(self, name: str) -> ScalerState:
        """
        Get a fitted scaler.

        Parameters
        ----------
        name : str
            The name of the scaler.

        Returns
        -------
        ScalerState
            The state of the scaler.

        Raises
        ------
        ValueError
            If the checkpoint has no scaler of that name.
        """
        state: ScalerState | None = self.scalers.get(name)
        if state is None:
            raise ValueError(f"checkpoint has no scaler {name!r}")
        return state

    def _read(self, name: str) -> EncodedValue:
        if name not in self.values:
            raise ValueError(f"checkpoint has no value {name!r}")
        return self.values[name]

    @staticmethod
    def _encode_scaler(state: ScalerState) -> dict[str, object]:
        return {
            "strategy": state.strategy.value,
            "parameters": dict(state.parameters),
            "n_features": state.n_features,
            # A copy, since PyTorch warns about sharing the memory of read-only arrays.
            "arrays": {name: torch.from_numpy(np.array(array)) for name, array in state.arrays.items()},
            "scikit_learn_version": state.scikit_learn_version,
            "empty_attributes": list(state.empty_attributes),
        }

    @staticmethod
    def _decode_scaler(encoded: object, location: str) -> ScalerState:
        state: dict[str, object] = Checkpoint._require_dictionary(encoded, location)
        strategy: object = state.get("strategy")
        parameters: EncodedValue = Checkpoint._require_encoded(state.get("parameters"), f"{location}.parameters")
        n_features: object = state.get("n_features")
        arrays: dict[str, object] = Checkpoint._require_dictionary(state.get("arrays"), f"{location}.arrays")
        scikit_learn_version: object = state.get("scikit_learn_version")
        empty_attributes: EncodedValue = Checkpoint._require_encoded(
            state.get("empty_attributes"), f"{location}.empty_attributes"
        )
        if (
            not isinstance(strategy, str)
            or not isinstance(parameters, dict)
            or not isinstance(n_features, int)
            or isinstance(n_features, bool)
            or not isinstance(scikit_learn_version, str)
            or not isinstance(empty_attributes, list)
            or not all(isinstance(name, str) for name in empty_attributes)
        ):
            raise ValueError(f"{location}: invalid scaler state")
        array_values: dict[str, NDArray[np.generic]] = {
            name: Checkpoint._require_tensor(tensor, f"{location}.arrays.{name}").numpy()
            for name, tensor in arrays.items()
        }
        return ScalerState(
            strategy=ScalingStrategy(strategy),
            parameters=parameters,
            n_features=n_features,
            arrays=array_values,
            scikit_learn_version=scikit_learn_version,
            empty_attributes=tuple(str(name) for name in empty_attributes),
        )

    @staticmethod
    def _require_dictionary(value: object, location: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected a mapping, got {type(value).__name__}")
        dictionary: dict[object, object] = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in dictionary):
            raise ValueError(f"{location}: expected string keys")
        return {str(key): item for key, item in dictionary.items()}

    @staticmethod
    def _require_tensor(value: object, location: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise ValueError(f"{location}: expected a tensor, got {type(value).__name__}")
        return value

    @staticmethod
    def _require_encoded(value: object, location: str) -> EncodedValue:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, list):
            items: list[object] = cast(list[object], value)
            return [Checkpoint._require_encoded(item, f"{location}[{index}]") for index, item in enumerate(items)]
        dictionary: dict[str, object] = Checkpoint._require_dictionary(value, location)
        return {key: Checkpoint._require_encoded(item, f"{location}.{key}") for key, item in dictionary.items()}
