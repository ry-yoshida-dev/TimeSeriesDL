import importlib.util
from pathlib import Path
from typing import Final

from .missing_dependency_error import MissingDependencyError
from .weights import PretrainedWeights

# Replaces the default user agent of huggingface_hub, which reports the installed torch version and any detected AI
# agent harness; only this fixed string is sent with the requests.
_USER_AGENT: Final[str] = "time-series-dl"


class HubClient:
    """
    Downloads published weight files from the Hugging Face Hub into its local cache.

    ``huggingface_hub`` is an optional dependency (the ``pretrained`` extra) and is imported only when a file is
    requested. Files are fetched from ``https://huggingface.co`` (or ``HF_ENDPOINT``) and its storage backend, and
    reused from the cache (``HF_HOME``, by default ``~/.cache/huggingface``) afterwards; with ``HF_HUB_OFFLINE=1``
    only the cache is read. Requests carry the fixed user agent ``time-series-dl`` instead of the library's default,
    which would list the installed torch version and any detected agent harness, and the access token stored by
    ``hf auth login``, if any, as huggingface_hub does by default (needed only for gated repositories). The download
    function sends no other telemetry.
    """

    @staticmethod
    def download(weights: PretrainedWeights) -> Path:
        """
        Get the local path of a weight file, downloading it if it is not cached.

        Parameters
        ----------
        weights : PretrainedWeights
            The repository, revision and file.

        Returns
        -------
        Path
            The path of the cached file.

        Raises
        ------
        MissingDependencyError
            If ``huggingface_hub`` is not installed.
        """
        if importlib.util.find_spec("huggingface_hub") is None:
            raise MissingDependencyError(
                "downloading published weights needs huggingface_hub; install the extra: "
                + "pip install 'time-series-dl[pretrained]'"
            )
        import huggingface_hub

        path: str = huggingface_hub.hf_hub_download(
            repo_id=weights.repository,
            filename=weights.filename,
            revision=weights.revision,
            headers={"user-agent": _USER_AGENT},
        )
        return Path(path)
