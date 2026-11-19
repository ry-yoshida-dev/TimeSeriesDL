import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

_REPOSITORY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._/-]+")


@dataclass(frozen=True, kw_only=True)
class PretrainedWeights:
    """
    Published weights of a model in a Hugging Face Hub repository, stored as a safetensors file.

    Attributes
    ----------
    repository : str
        Repository id in the form ``owner/name``, such as ``amazon/chronos-bolt-small``.
    revision : str
        Commit hash, branch or tag. A commit hash pins the exact files, so the weights cannot change unnoticed.
    filename : str
        Path of the safetensors file inside the repository.

    Raises
    ------
    ValueError
        If the repository id, the revision or the file name is malformed, or the file is not a ``.safetensors`` file
        inside the repository.
    """

    repository: str
    revision: str
    filename: str = "model.safetensors"

    def __post_init__(self) -> None:
        if _REPOSITORY_PATTERN.fullmatch(self.repository) is None:
            raise ValueError(f"repository must have the form 'owner/name', got {self.repository!r}")
        if _REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError(f"revision must be a commit hash, branch or tag, got {self.revision!r}")
        path: PurePosixPath = PurePosixPath(self.filename)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            raise ValueError(f"filename must be a relative .safetensors path, got {self.filename!r}")
