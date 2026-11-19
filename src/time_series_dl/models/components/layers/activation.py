from enum import Enum

from torch import nn


class Activation(Enum):
    """
    Element-wise activation function.

    Attributes
    ----------
    RELU
        Rectified linear unit.
    GELU
        Gaussian error linear unit.
    SILU
        Sigmoid linear unit (swish).
    TANH
        Hyperbolic tangent.
    """

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"

    def build(self) -> nn.Module:
        """
        Build the activation module.

        Returns
        -------
        nn.Module
            The activation.
        """
        match self:
            case Activation.RELU:
                return nn.ReLU()
            case Activation.GELU:
                return nn.GELU()
            case Activation.SILU:
                return nn.SiLU()
            case Activation.TANH:
                return nn.Tanh()
