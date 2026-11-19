from dataclasses import dataclass

from ....components.random_convolution import KernelTransformSummaryNetwork
from ....interfaces import SummaryNetwork, SummaryShape
from .kernel_transform import RocketKernelTransform


@dataclass(frozen=True, kw_only=True)
class RocketParameters:
    """
    ROCKET (Dempster et al., 2020): random convolution kernels whose PPV and max features feed a linear classifier.

    The kernels are drawn once when the network is built and kept fixed; only a linear layer on the standardized
    features is trained, by gradient descent with the task's loss instead of the paper's ridge regression. The
    defaults are those of the paper.

    Attributes
    ----------
    num_kernels : int
        Number of random kernels; each gives two features.
    kernel_lengths : tuple[int, ...]
        Kernel lengths drawn uniformly; odd, so padded kernels keep the length.
    is_normalized : bool
        Whether each channel of each sequence is standardized before the convolutions, as the uniform biases in
        [-1, 1] assume.

    Raises
    ------
    ValueError
        If num_kernels is not positive, or kernel_lengths is empty or holds a length that is not a positive odd number.
    """

    num_kernels: int = 10_000
    kernel_lengths: tuple[int, ...] = (7, 9, 11)
    is_normalized: bool = True

    def __post_init__(self) -> None:
        if self.num_kernels <= 0:
            raise ValueError(f"num_kernels must be positive, got {self.num_kernels}")
        if not self.kernel_lengths:
            raise ValueError("kernel_lengths must not be empty.")
        if any(length <= 0 or length % 2 == 0 for length in self.kernel_lengths):
            raise ValueError(f"kernel_lengths must be positive odd numbers, got {self.kernel_lengths}")

    def build_transform(self, input_length: int, input_channels: int) -> RocketKernelTransform:
        """
        Draw the random kernels.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        RocketKernelTransform
            The transform.
        """
        return RocketKernelTransform(self, input_length, input_channels)

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build an untrained summary network.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.
        """
        return KernelTransformSummaryNetwork(self.build_transform(shape.input_length, shape.input_channels), shape)
