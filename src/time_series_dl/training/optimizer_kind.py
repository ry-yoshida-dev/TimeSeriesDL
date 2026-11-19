from collections.abc import Iterable
from enum import Enum

import torch
from torch import nn


class OptimizerKind(Enum):
    """
    Gradient-based optimizer.

    Attributes
    ----------
    ADAM
        Adam.
    ADAMW
        Adam with decoupled weight decay.
    SGD
        Stochastic gradient descent with Nesterov momentum 0.9.
    """

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

    def build(
        self, parameters: Iterable[nn.Parameter], learning_rate: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        Build the optimizer of the given parameters.

        Parameters
        ----------
        parameters : Iterable[nn.Parameter]
            The trainable parameters.
        learning_rate : float
            The initial learning rate.
        weight_decay : float
            The weight decay coefficient.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        match self:
            case OptimizerKind.ADAM:
                return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
            case OptimizerKind.ADAMW:
                return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
            case OptimizerKind.SGD:
                return torch.optim.SGD(
                    parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True
                )
