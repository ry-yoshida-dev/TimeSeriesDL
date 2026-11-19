import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NameRule:
    """
    Renaming of published parameter names: every match of a regular expression is replaced.

    Attributes
    ----------
    pattern : str
        Regular expression matched against the name.
    replacement : str
        Replacement, which may refer to groups of the pattern (``\\1``).

    Raises
    ------
    ValueError
        If the pattern is empty or not a valid regular expression.
    """

    pattern: str
    replacement: str
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern must not be empty.")
        try:
            compiled: re.Pattern[str] = re.compile(self.pattern)
        except re.error as error:
            raise ValueError(f"pattern {self.pattern!r} is not a valid regular expression: {error}") from error
        object.__setattr__(self, "_compiled", compiled)

    def apply(self, name: str) -> str:
        """
        Rename one parameter.

        Parameters
        ----------
        name : str
            The name before this rule.

        Returns
        -------
        str
            The name with every match replaced.
        """
        return self._compiled.sub(self.replacement, name)
