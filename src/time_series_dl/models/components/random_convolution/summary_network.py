from typing import override

from torch import Tensor, nn

from ...interfaces import SummaryNetwork, SummaryShape
from .kernel_transform import KernelTransform


class KernelTransformSummaryNetwork(SummaryNetwork):
    """
    Fixed kernel transform followed by feature standardization and a trained linear layer.

    The papers fit a ridge or logistic regression on standardized features; here the linear layer is trained by
    gradient descent with the task's loss, which is the logistic regression variant for cross-entropy. Features are
    standardized by batch normalization without affine parameters, so the linear layer is the only trained part.

    Parameters
    ----------
    transform : KernelTransform
        The fixed transform of the sequences.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, transform: KernelTransform, shape: SummaryShape) -> None:
        super().__init__()
        self.transform: KernelTransform = transform
        self.standardization: nn.BatchNorm1d = nn.BatchNorm1d(transform.num_features, affine=False)
        self.projection: nn.Linear = nn.Linear(transform.num_features, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.standardization(self.transform(values))
        outputs: Tensor = self.projection(features)
        return outputs
