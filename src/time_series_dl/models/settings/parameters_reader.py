from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import cast, get_args, get_origin, get_type_hints


class ParametersReader:
    """
    Builds a parameters dataclass from plain configuration values, such as a section of a YAML file.

    Enums are given by value, nested dataclasses as mappings and tuples as sequences, recursively; fields left out
    keep their defaults.
    """

    @staticmethod
    def read[T](parameters_type: type[T], values: Mapping[str, object]) -> T:
        """
        Build ``parameters_type`` from ``values``.

        Parameters
        ----------
        parameters_type : type[T]
            Dataclass to build.
        values : Mapping[str, object]
            Field values by field name.

        Returns
        -------
        T
            The parameters.

        Raises
        ------
        TypeError
            If ``parameters_type`` is not a dataclass or a value has the wrong type.
        KeyError
            If a key is not a field of ``parameters_type``.
        ValueError
            If a value is not a member of its enum, or the parameters reject a value.
        """
        if not is_dataclass(parameters_type):
            raise TypeError(f"{parameters_type} is not a dataclass.")
        field_names: set[str] = {field.name for field in fields(parameters_type) if field.init}
        unknown_names: set[str] = set(values) - field_names
        if unknown_names:
            raise KeyError(f"{parameters_type.__name__} has no fields {sorted(unknown_names)}")
        annotations: dict[str, object] = get_type_hints(parameters_type)
        arguments: dict[str, object] = {
            name: ParametersReader._convert(annotations[name], value, f"{parameters_type.__name__}.{name}")
            for name, value in values.items()
        }
        return parameters_type(**arguments)

    @staticmethod
    def _convert(annotation: object, value: object, location: str) -> object:
        origin: object = get_origin(annotation)
        if origin is UnionType:
            options: tuple[object, ...] = get_args(annotation)
            if value is None and NoneType in options:
                return None
            non_none_options: list[object] = [option for option in options if option is not NoneType]
            if len(non_none_options) != 1:
                # Values that are already built, such as decoded nested settings, are kept when one option fits.
                if any(isinstance(option, type) and isinstance(value, option) for option in non_none_options):
                    return value
                raise TypeError(f"{location}: unions of several types are not supported. got {annotation}")
            return ParametersReader._convert(non_none_options[0], value, location)
        if origin is tuple:
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise TypeError(f"{location}: expected a sequence. got {type(value).__name__}")
            # isinstance cannot check item types; the conversion below checks every item.
            items: list[object] = list(cast(Sequence[object], value))
            item_types: tuple[object, ...] = get_args(annotation)
            if len(item_types) == 2 and item_types[1] is Ellipsis:
                item_types = (item_types[0],) * len(items)
            if len(item_types) != len(items):
                raise TypeError(f"{location}: expected {len(item_types)} items. got {len(items)}")
            return tuple(
                ParametersReader._convert(item_type, item, f"{location}[{index}]")
                for index, (item_type, item) in enumerate(zip(item_types, items, strict=True))
            )
        if not isinstance(annotation, type):
            raise TypeError(f"{location}: unsupported annotation {annotation}")
        annotation_type: type[object] = annotation
        if issubclass(annotation_type, Enum):
            enum_type: type[Enum] = annotation_type
            return value if isinstance(value, enum_type) else enum_type(value)
        if is_dataclass(annotation_type):
            if isinstance(value, annotation_type):
                return value
            if not isinstance(value, Mapping):
                raise TypeError(f"{location}: expected a mapping. got {type(value).__name__}")
            mapping: Mapping[object, object] = cast(Mapping[object, object], value)
            return ParametersReader.read(annotation_type, {str(key): item for key, item in mapping.items()})
        # YAML writes whole floats such as 1.0 as integers when they are hand-edited to 1.
        if annotation_type is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if not isinstance(value, annotation_type) or (annotation_type is int and isinstance(value, bool)):
            raise TypeError(f"{location}: expected {annotation_type.__name__}. got {type(value).__name__}")
        return value
