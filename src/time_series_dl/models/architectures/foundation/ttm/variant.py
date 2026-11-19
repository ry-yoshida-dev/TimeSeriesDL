from enum import Enum


class TTMVariant(Enum):
    """
    Published TinyTimeMixer r2 checkpoint (``ibm-granite/granite-timeseries-ttm-r2``, Apache-2.0), by context
    length and prediction length; each is a branch of the repository.

    Attributes
    ----------
    CONTEXT_512_PREDICTION_96
        The ``main`` branch: 512 past values, 96 forecast values.
    CONTEXT_512_PREDICTION_192
        Branch ``512-192-r2``.
    CONTEXT_512_PREDICTION_336
        Branch ``512-336-r2``.
    CONTEXT_512_PREDICTION_720
        Branch ``512-720-r2``.
    CONTEXT_1024_PREDICTION_96
        Branch ``1024-96-r2``.
    CONTEXT_1024_PREDICTION_192
        Branch ``1024-192-r2``.
    CONTEXT_1024_PREDICTION_336
        Branch ``1024-336-r2``.
    CONTEXT_1024_PREDICTION_720
        Branch ``1024-720-r2``.
    CONTEXT_1536_PREDICTION_96
        Branch ``1536-96-r2``.
    CONTEXT_1536_PREDICTION_192
        Branch ``1536-192-r2``.
    CONTEXT_1536_PREDICTION_336
        Branch ``1536-336-r2``.
    CONTEXT_1536_PREDICTION_720
        Branch ``1536-720-r2``.
    """

    CONTEXT_512_PREDICTION_96 = "512-96-r2"
    CONTEXT_512_PREDICTION_192 = "512-192-r2"
    CONTEXT_512_PREDICTION_336 = "512-336-r2"
    CONTEXT_512_PREDICTION_720 = "512-720-r2"
    CONTEXT_1024_PREDICTION_96 = "1024-96-r2"
    CONTEXT_1024_PREDICTION_192 = "1024-192-r2"
    CONTEXT_1024_PREDICTION_336 = "1024-336-r2"
    CONTEXT_1024_PREDICTION_720 = "1024-720-r2"
    CONTEXT_1536_PREDICTION_96 = "1536-96-r2"
    CONTEXT_1536_PREDICTION_192 = "1536-192-r2"
    CONTEXT_1536_PREDICTION_336 = "1536-336-r2"
    CONTEXT_1536_PREDICTION_720 = "1536-720-r2"

    @property
    def repository(self) -> str:
        """
        Get the Hugging Face Hub repository of the checkpoint.

        Returns
        -------
        str
            The repository id.
        """
        return "ibm-granite/granite-timeseries-ttm-r2"

    @property
    def context_length(self) -> int:
        """
        Get the number of past values the checkpoint reads.

        Returns
        -------
        int
            The context length.
        """
        return int(self.value.split("-")[0])

    @property
    def prediction_length(self) -> int:
        """
        Get the number of future values the checkpoint forecasts.

        Returns
        -------
        int
            The prediction length.
        """
        return int(self.value.split("-")[1])

    @property
    def revision(self) -> str:
        """
        Get the commit of the repository the checkpoint is pinned to.

        Returns
        -------
        str
            The commit hash of the checkpoint's branch.
        """
        match self:
            case TTMVariant.CONTEXT_512_PREDICTION_96:
                return "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
            case TTMVariant.CONTEXT_512_PREDICTION_192:
                return "25f4a00a25e1d0b67c40a10b06645d72e2266929"
            case TTMVariant.CONTEXT_512_PREDICTION_336:
                return "0009f51f52c87473e60ff3a03cd8633ee85d61e8"
            case TTMVariant.CONTEXT_512_PREDICTION_720:
                return "bcfc4a497430a228afcbaf378f18f316642ece2a"
            case TTMVariant.CONTEXT_1024_PREDICTION_96:
                return "1a382618f3c5b63836281ad632e54724e55e85a5"
            case TTMVariant.CONTEXT_1024_PREDICTION_192:
                return "c6255e0d68d795a121d10c0889bdfc8684319076"
            case TTMVariant.CONTEXT_1024_PREDICTION_336:
                return "631e9f15b5cf8f5f29f83b6a72952b4d19d1d15a"
            case TTMVariant.CONTEXT_1024_PREDICTION_720:
                return "3a31783dd34f7d051e3e0ff2379435f4c0604cad"
            case TTMVariant.CONTEXT_1536_PREDICTION_96:
                return "f6ed09d67cf42e4877f7fa33d98326911385693d"
            case TTMVariant.CONTEXT_1536_PREDICTION_192:
                return "ee7a2930c0f7e3ad9860e41174e8a6c936af1328"
            case TTMVariant.CONTEXT_1536_PREDICTION_336:
                return "f34db3be503935a15a0381963eeeda0b0bedd309"
            case TTMVariant.CONTEXT_1536_PREDICTION_720:
                return "3ca910a81018c6a6159a02e2505f3d62c39e91a4"
