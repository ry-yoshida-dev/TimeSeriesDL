from enum import Enum

from torch import nn


class CycleNetBackbone(Enum):
    """
    Network of CycleNet that maps each channel's residual window to the residual horizon.

    Attributes
    ----------
    LINEAR
        One linear layer (CycleNet/Linear).
    MLP
        Two linear layers with a ReLU between them (CycleNet/MLP).
    """

    LINEAR = "linear"
    MLP = "mlp"

    def build(self, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
        """
        Build the untrained backbone.

        Parameters
        ----------
        input_size : int
            Number of input features (the window length).
        hidden_size : int
            Size of the hidden layer; unused by the linear backbone.
        output_size : int
            Number of output features.

        Returns
        -------
        nn.Module
            The backbone, applied to the last dimension.
        """
        match self:
            case CycleNetBackbone.LINEAR:
                return nn.Linear(input_size, output_size)
            case CycleNetBackbone.MLP:
                return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
