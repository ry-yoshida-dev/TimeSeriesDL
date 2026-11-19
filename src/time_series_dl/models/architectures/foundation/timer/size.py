from enum import Enum


class TimerSize(Enum):
    """
    Published Timer checkpoint (Apache-2.0), by model size.

    Attributes
    ----------
    BASE
        ``thuml/timer-base-84m``, 84M parameters, pretrained on 260B time points.
    """

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
        match self:
            case TimerSize.BASE:
                return "thuml/timer-base-84m"

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
            case TimerSize.BASE:
                return "70077a71acce1b4c00d98332fcaabc694255d8e5"
