import importlib.util
from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from .missing_dependency_error import MissingDependencyError


class SafetensorsFile:
    """
    Reads and writes weight files in the safetensors format, which stores raw tensors and runs no code on loading.

    ``safetensors`` is an optional dependency (the ``pretrained`` extra) and is imported only when a file is read or
    written.
    """

    @staticmethod
    def read(path: Path) -> dict[str, Tensor]:
        """
        Read every tensor of a file onto the CPU.

        Parameters
        ----------
        path : Path
            The file.

        Returns
        -------
        dict[str, Tensor]
            The tensors by name.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        MissingDependencyError
            If ``safetensors`` is not installed.
        """
        if not path.is_file():
            raise FileNotFoundError(f"weight file {path} does not exist.")
        SafetensorsFile._require_library()
        import safetensors.torch

        return safetensors.torch.load_file(path, device="cpu")

    @staticmethod
    def write(path: Path, tensors: Mapping[str, Tensor]) -> None:
        """
        Write tensors to a file.

        Parameters
        ----------
        path : Path
            The file to write.
        tensors : Mapping[str, Tensor]
            The tensors by name; each is written contiguously.

        Raises
        ------
        MissingDependencyError
            If ``safetensors`` is not installed.
        """
        SafetensorsFile._require_library()
        import safetensors.torch

        safetensors.torch.save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, path)

    @staticmethod
    def _require_library() -> None:
        if importlib.util.find_spec("safetensors") is None:
            raise MissingDependencyError(
                "reading published weights needs safetensors; install the extra: "
                + "pip install 'time-series-dl[pretrained]'"
            )
