from enum import Enum


class ChronosBoltSize(Enum):
    """
    Published Chronos-Bolt checkpoint (Apache-2.0), by model size.

    Attributes
    ----------
    TINY
        ``amazon/chronos-bolt-tiny``, 9M parameters.
    MINI
        ``amazon/chronos-bolt-mini``, 21M parameters.
    SMALL
        ``amazon/chronos-bolt-small``, 48M parameters.
    BASE
        ``amazon/chronos-bolt-base``, 205M parameters.
    """

    TINY = "tiny"
    MINI = "mini"
    SMALL = "small"
    BASE = "base"

    @property
    def repository(self) -> str:
        """
        Get the Hugging Face Hub repository of the checkpoint.

        Returns
        -------
        str
            The repository id.
        """
        return f"amazon/chronos-bolt-{self.value}"

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
            case ChronosBoltSize.TINY:
                return "a0e552de83495b5c28c14c71c374f3e33280b340"
            case ChronosBoltSize.MINI:
                return "251268337516a88e253628c43e1d26ec577b376b"
            case ChronosBoltSize.SMALL:
                return "772f3d25d38aec6d914c8949dab4462e2d46f5d8"
            case ChronosBoltSize.BASE:
                return "5d9f166d69f47aef3401367a7b842e78fe97b121"
