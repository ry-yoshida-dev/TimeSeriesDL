from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Final, cast, get_args, get_type_hints

from data_scaler import (
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    NoScalingParameters,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
)
from deep_learning_loss import (
    CharbonnierLoss,
    CrossEntropyLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MAPELoss,
    MSELoss,
    MSLELoss,
    MulticlassHingeLoss,
    NTXentLoss,
    PinballLoss,
    Poly1Loss,
    QuantileLoss,
    SMAPELoss,
)

from ..models import ModelArchitecture
from ..models.settings.parameters_reader import ParametersReader
from ..training import TrainingHistory
from .encoded_value import EncodedValue

_TYPE_KEY: Final[str] = "type"
_FIELDS_KEY: Final[str] = "fields"

# Types every task can hold: the parameters of every model, the losses of DeepLearningLoss that a task or model
# accepts, the scaling strategies and the history. Each task adds its settings class and the implementations of its
# own abstract fields.
_COMMON_TYPES: Final[tuple[type[object], ...]] = (
    *(architecture.parameters_type for architecture in ModelArchitecture),
    CharbonnierLoss,
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MAPELoss,
    MSELoss,
    MSLELoss,
    PinballLoss,
    SMAPELoss,
    QuantileLoss,
    CrossEntropyLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    MulticlassHingeLoss,
    Poly1Loss,
    NTXentLoss,
    MaxAbsScalingParameters,
    MinMaxScalingParameters,
    NoScalingParameters,
    PowerScalingParameters,
    QuantileScalingParameters,
    RobustScalingParameters,
    StandardScalingParameters,
    UnitNormScalingParameters,
    TrainingHistory,
)


@dataclass(frozen=True)
class SettingsCodec:
    """
    Converts settings dataclasses to and from plain values, so they are saved without pickle.

    Each dataclass is written with the name of its class, which is looked up among ``types`` when it is read back,
    so fields typed by an abstract class or a protocol, such as the model or the loss, keep their implementation.
    Only the listed classes are ever built, and every value is validated by the class it belongs to.

    Attributes
    ----------
    types : Mapping[str, type[object]]
        The concrete dataclasses that may be read, by class name.
    """

    types: Mapping[str, type[object]]

    @staticmethod
    def including(*roots: type[object]) -> SettingsCodec:
        """
        Build a codec of the given dataclasses, the types every task shares, and every dataclass their fields name.

        Parameters
        ----------
        *roots : type[object]
            Settings classes and the implementations of their abstract fields.

        Returns
        -------
        SettingsCodec
            The codec.

        Raises
        ------
        ValueError
            If two different dataclasses share a name.
        """
        found: dict[str, type[object]] = {}
        pending: list[object] = [*_COMMON_TYPES, *roots]
        while pending:
            annotation: object = pending.pop()
            arguments: tuple[object, ...] = get_args(annotation)
            if arguments:
                pending.extend(arguments)
                continue
            if not isinstance(annotation, type) or not is_dataclass(annotation):
                continue
            name: str = annotation.__name__
            known: type[object] | None = found.get(name)
            if known is not None:
                if known is not annotation:
                    raise ValueError(f"{known.__module__}.{name} and {annotation.__module__}.{name} share a name.")
                continue
            if not inspect.isabstract(annotation):
                found[name] = annotation
            pending.extend(get_type_hints(annotation).values())
        return SettingsCodec(found)

    def encode(self, value: object) -> EncodedValue:
        """
        Convert a value to plain values.

        Parameters
        ----------
        value : object
            A dataclass among ``types``, or a field value of one: enums are written by value and tuples as lists.

        Returns
        -------
        EncodedValue
            The plain value.

        Raises
        ------
        TypeError
            If a dataclass is not among ``types`` or a value has an unsupported type.
        """
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, Enum):
            return self.encode(cast(object, value.value))
        if isinstance(value, tuple):
            items: tuple[object, ...] = cast(tuple[object, ...], value)
            return [self.encode(item) for item in items]
        if is_dataclass(value) and not isinstance(value, type):
            name: str = type(value).__name__
            if self.types.get(name) is not type(value):
                raise TypeError(f"{type(value).__qualname__} cannot be saved; it is not a known settings class.")
            encoded_fields: dict[str, EncodedValue] = {
                field.name: self.encode(cast(object, getattr(value, field.name)))
                for field in fields(value)
                if field.init
            }
            return {_TYPE_KEY: name, _FIELDS_KEY: encoded_fields}
        raise TypeError(f"values of type {type(value).__name__} cannot be saved.")

    def decode[T](self, expected_type: type[T], encoded: EncodedValue) -> T:
        """
        Rebuild a dataclass written by encode.

        Parameters
        ----------
        expected_type : type[T]
            The class, abstract class or protocol the value must be an instance of.
        encoded : EncodedValue
            The plain value.

        Returns
        -------
        T
            The value.

        Raises
        ------
        ValueError
            If the value names an unknown class, is malformed, or is rejected by its class.
        TypeError
            If the value is not an instance of ``expected_type`` or a field has the wrong type.
        KeyError
            If a field is not a field of its class.
        """
        value: object = self._decode(encoded, expected_type.__name__)
        if not isinstance(value, expected_type):
            raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
        return value

    def _decode(self, encoded: EncodedValue, location: str) -> object:
        if isinstance(encoded, list):
            return [self._decode(item, f"{location}[{index}]") for index, item in enumerate(encoded)]
        if not isinstance(encoded, dict):
            return encoded
        name: EncodedValue = encoded.get(_TYPE_KEY)
        encoded_fields: EncodedValue = encoded.get(_FIELDS_KEY)
        if not isinstance(name, str) or not isinstance(encoded_fields, dict) or len(encoded) != 2:
            raise ValueError(f"{location}: expected a mapping of {_TYPE_KEY!r} and {_FIELDS_KEY!r}")
        dataclass_type: type[object] | None = self.types.get(name)
        if dataclass_type is None:
            raise ValueError(f"{location}: unknown settings class {name!r}")
        # Nested dataclasses are built first; the reader then checks every field against its annotation.
        values: dict[str, object] = {
            field_name: self._decode(item, f"{name}.{field_name}") for field_name, item in encoded_fields.items()
        }
        return ParametersReader.read(dataclass_type, values)
