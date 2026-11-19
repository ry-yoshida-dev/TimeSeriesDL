from dataclasses import dataclass

from ....components.random_convolution import KernelPooling, KernelTransformSummaryNetwork
from ....interfaces import SummaryNetwork, SummaryShape
from .kernel_transform import MiniRocketKernelTransform

_NUM_KERNELS: int = 84


@dataclass(frozen=True, kw_only=True)
class MiniRocketParameters:
    """
    MiniRocket (Dempster et al., 2021): 84 fixed kernels at several dilations, each with several biases, pooled into
    their proportions of positive values (PPV) for a linear classifier.

    The biases are fitted to the first batch the network sees and then kept fixed; only a linear layer on the
    standardized features is trained, by gradient descent with the task's loss instead of the paper's ridge
    regression. The defaults are those of the paper.

    Attributes
    ----------
    num_features : int
        Target number of features; rounded down to a multiple of 84.
    max_dilations_per_kernel : int
        Largest number of dilations per kernel.

    Raises
    ------
    ValueError
        If num_features is below 84 or max_dilations_per_kernel is not positive.
    """

    num_features: int = 10_000
    max_dilations_per_kernel: int = 32

    def __post_init__(self) -> None:
        if self.num_features < _NUM_KERNELS:
            raise ValueError(f"num_features must be at least {_NUM_KERNELS}, got {self.num_features}")
        if self.max_dilations_per_kernel <= 0:
            raise ValueError(f"max_dilations_per_kernel must be positive, got {self.max_dilations_per_kernel}")

    def build_transform(self, input_length: int, input_channels: int) -> MiniRocketKernelTransform:
        """
        Build the transform with unfitted biases.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        MiniRocketKernelTransform
            The transform.
        """
        return MiniRocketKernelTransform(
            input_length,
            input_channels,
            self.num_features,
            self.max_dilations_per_kernel,
            (KernelPooling.PROPORTION_OF_POSITIVE_VALUES,),
        )

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
