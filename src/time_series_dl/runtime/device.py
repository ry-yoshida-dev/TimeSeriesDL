from enum import Enum

import torch


class Device(Enum):
    """
    Device that networks are trained and run on.

    Attributes
    ----------
    AUTO
        CUDA if available, then Apple MPS, then the CPU.
    CPU
        The CPU.
    CUDA
        The default CUDA GPU.
    MPS
        Apple Metal Performance Shaders.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    def resolve(self) -> torch.device:
        """
        Get the PyTorch device of this choice.

        Returns
        -------
        torch.device
            The device.

        Raises
        ------
        RuntimeError
            If the requested accelerator is not available.
        """
        match self:
            case Device.AUTO:
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            case Device.CPU:
                return torch.device("cpu")
            case Device.CUDA:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available on this machine.")
                return torch.device("cuda")
            case Device.MPS:
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS is not available on this machine.")
                return torch.device("mps")
