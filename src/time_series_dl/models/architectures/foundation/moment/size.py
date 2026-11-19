from enum import Enum


class MOMENTSize(Enum):
    """
    Published MOMENT checkpoint (MIT license), by model size.

    Attributes
    ----------
    SMALL
        ``AutonLab/MOMENT-1-small``: a Flan-T5-small encoder, 38M parameters.
    BASE
        ``AutonLab/MOMENT-1-base``: a Flan-T5-base encoder, 113M parameters.
    LARGE
        ``AutonLab/MOMENT-1-large``: a Flan-T5-large encoder, 346M parameters.
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
        return f"AutonLab/MOMENT-1-{self.value}"

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
            case MOMENTSize.SMALL:
                return "411e288267f82cce86296dbe4d6c8bc533cc162f"
            case MOMENTSize.BASE:
                return "5e44b0ea26376a176360f87831124e018f876d96"
            case MOMENTSize.LARGE:
                return "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
