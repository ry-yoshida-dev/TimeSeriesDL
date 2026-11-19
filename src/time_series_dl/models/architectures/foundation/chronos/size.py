from enum import Enum


class ChronosSize(Enum):
    """
    Published Chronos checkpoint (Apache-2.0), by model size.

    Attributes
    ----------
    TINY
        ``amazon/chronos-t5-tiny``, 8M parameters.
    MINI
        ``amazon/chronos-t5-mini``, 20M parameters.
    SMALL
        ``amazon/chronos-t5-small``, 46M parameters.
    BASE
        ``amazon/chronos-t5-base``, 200M parameters.
    LARGE
        ``amazon/chronos-t5-large``, 710M parameters.
    """

    TINY = "tiny"
    MINI = "mini"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"

    @property
    def repository(self) -> str:
        """
        Get the Hugging Face Hub repository of the checkpoint.

        Returns
        -------
        str
            The repository id.
        """
        return f"amazon/chronos-t5-{self.value}"

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
            case ChronosSize.TINY:
                return "29d808298f1a62493e7b9a5e08529d0d930fa189"
            case ChronosSize.MINI:
                return "bd6a4fde8403b8469acd0abd52852b29dbe61c7b"
            case ChronosSize.SMALL:
                return "a971ba21945c4f1796b17a91fe69214b5f4ad472"
            case ChronosSize.BASE:
                return "ad294eaacead15db499b740ea4122266dd2a81a2"
            case ChronosSize.LARGE:
                return "0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a"
