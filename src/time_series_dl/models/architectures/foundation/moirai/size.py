from enum import Enum


class MoiraiSize(Enum):
    """
    Published Moirai 1.1-R checkpoint, by model size. The weights are licensed CC-BY-NC-4.0: non-commercial use only.

    Attributes
    ----------
    SMALL
        ``Salesforce/moirai-1.1-R-small``, 14M parameters.
    BASE
        ``Salesforce/moirai-1.1-R-base``, 91M parameters.
    LARGE
        ``Salesforce/moirai-1.1-R-large``, 311M parameters.
    """

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
        return f"Salesforce/moirai-1.1-R-{self.value}"

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
            case MoiraiSize.SMALL:
                return "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
            case MoiraiSize.BASE:
                return "a0358780de4cdd7ac7963b12ae9985d225d55026"
            case MoiraiSize.LARGE:
                return "d144331c5a18e986a9828e7378994b39e209d466"
