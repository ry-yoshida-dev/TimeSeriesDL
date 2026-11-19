import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor

from .name_rule import NameRule
from .weights_mismatch_error import WeightsMismatchError


@dataclass(frozen=True)
class ParameterNameMapping:
    """
    Translation of the parameter names of published weights into the names of a network's state dict.

    Each published name passes through every rule in order; names matching an ignored pattern are dropped first,
    which is how weights the network deliberately does not have (such as an unused pretraining head) are skipped.

    Attributes
    ----------
    rules : tuple[NameRule, ...]
        Renamings applied in order.
    ignored : tuple[str, ...]
        Regular expressions of published names that are not loaded; matched with ``re.fullmatch``.

    Raises
    ------
    ValueError
        If an ignored pattern is not a valid regular expression.
    """

    rules: tuple[NameRule, ...] = ()
    ignored: tuple[str, ...] = ()
    _ignored_patterns: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        patterns: list[re.Pattern[str]] = []
        for pattern in self.ignored:
            try:
                patterns.append(re.compile(pattern))
            except re.error as error:
                raise ValueError(f"ignored pattern {pattern!r} is not a valid regular expression: {error}") from error
        object.__setattr__(self, "_ignored_patterns", tuple(patterns))

    def rename(self, name: str) -> str:
        """
        Translate one published name.

        Parameters
        ----------
        name : str
            The published name.

        Returns
        -------
        str
            The name in the network's state dict.
        """
        for rule in self.rules:
            name = rule.apply(name)
        return name

    def is_ignored(self, name: str) -> bool:
        """
        Check whether a published name is deliberately not loaded.

        Parameters
        ----------
        name : str
            The published name.

        Returns
        -------
        bool
            True if an ignored pattern matches the whole name.
        """
        return any(pattern.fullmatch(name) is not None for pattern in self._ignored_patterns)

    def apply(self, tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """
        Translate every published name of a set of weights.

        Parameters
        ----------
        tensors : Mapping[str, Tensor]
            Published weights by published name.

        Returns
        -------
        dict[str, Tensor]
            The weights that are not ignored, by state dict name.

        Raises
        ------
        WeightsMismatchError
            If two published names translate to the same state dict name.
        """
        renamed: dict[str, Tensor] = {}
        sources: dict[str, str] = {}
        collisions: list[str] = []
        for name, tensor in tensors.items():
            if self.is_ignored(name):
                continue
            target: str = self.rename(name)
            if target in sources:
                collisions.append(f"{sources[target]} and {name} both map to {target}")
                continue
            sources[target] = name
            renamed[target] = tensor
        if collisions:
            raise WeightsMismatchError(missing=(), unexpected=collisions, mismatched=())
        return renamed
