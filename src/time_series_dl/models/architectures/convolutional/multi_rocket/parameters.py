from dataclasses import dataclass

from ....components.random_convolution import KernelTransformSummaryNetwork
from ....interfaces import SummaryNetwork, SummaryShape
from .kernel_transform import MultiRocketKernelTransform

_MIN_NUM_FEATURES: int = 84 * 2 * 4


@dataclass(frozen=True, kw_only=True)
class MultiRocketParameters:
    """
    MultiRocket (Tan et al., 2022): MiniRocket kernels on the series and its first difference, each output pooled
    into four features (PPV, MPV, MIPV and LSPV) for a linear classifier.

    The biases are fitted to the first batch the network sees and then kept fixed; only a linear layer on the
    standardized features is trained, by gradient descent with the task's loss instead of the paper's ridge
    regression. The defaults are those of the paper, giving 49,728 features.

    Attributes
    ----------
    num_features : int
        Target number of features; each of the two representations gets an eighth of it as MiniRocket biases,
        rounded down to a multiple of 84.
    max_dilations_per_kernel : int
        Largest number of dilations per kernel.

    Raises
    ------
    ValueError
        If num_features is below 672 or max_dilations_per_kernel is not positive.
    """

    num_features: int = 50_000
    max_dilations_per_kernel: int = 32

    def __post_init__(self) -> None:
        if self.num_features < _MIN_NUM_FEATURES:
            raise ValueError(f"num_features must be at least {_MIN_NUM_FEATURES}, got {self.num_features}")
        if self.max_dilations_per_kernel <= 0:
            raise ValueError(f"max_dilations_per_kernel must be positive, got {self.max_dilations_per_kernel}")

    def build_transform(self, input_length: int, input_channels: int) -> MultiRocketKernelTransform:
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
        MultiRocketKernelTransform
            The transform.
        """
        return MultiRocketKernelTransform(self, input_length, input_channels)

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
