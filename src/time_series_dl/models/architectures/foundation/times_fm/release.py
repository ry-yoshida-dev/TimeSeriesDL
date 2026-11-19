from enum import Enum


class TimesFMRelease(Enum):
    """
    Published TimesFM checkpoint with PyTorch weights (Apache-2.0).

    Attributes
    ----------
    V2_0_500M
        ``google/timesfm-2.0-500m-pytorch``: 500M parameters, 50 layers, context of up to 2048 values.
    """

    V2_0_500M = "2.0-500m"

    @property
    def repository(self) -> str:
        """
        Get the Hugging Face Hub repository of the checkpoint.

        Returns
        -------
        str
            The repository id.
        """
        return f"google/timesfm-{self.value}-pytorch"

    @property
    def revision(self) -> str:
        """
        Get the commit of the repository the checkpoint is pinned to.

        Returns
        -------
        str
            The commit hash.
        """
        match self:
            case TimesFMRelease.V2_0_500M:
                return "dc2443792ce5516872b89b37cf1bc058c3bf0c10"
