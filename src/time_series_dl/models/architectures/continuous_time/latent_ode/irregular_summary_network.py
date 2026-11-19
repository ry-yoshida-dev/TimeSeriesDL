from collections.abc import Callable
from typing import override

from torch import Tensor, nn

from ....interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....interfaces.irregular_tensors import IrregularTensors
from .generative_model import LatentODEGenerativeModel
from .latent_posterior import LatentPosterior


class LatentODESummaryNetwork(IrregularSummaryNetwork):
    """
    Latent ODE classifier or regressor: output layers read the initial latent state, drawn from the posterior while
    training and its mean otherwise. Training adds the negative evidence lower bound of the observed values to the
    weighted task loss, as in the paper.

    Parameters
    ----------
    model : LatentODEGenerativeModel
        The generative model and its recognition network.
    output_layers : nn.Module
        Maps latent states with shape (batch, latent_size) to outputs with shape (batch, output_size).
    output_loss_weight : float
        Weight of the task loss relative to the evidence lower bound.

    Raises
    ------
    ValueError
        If output_loss_weight is not positive.
    """

    def __init__(self, model: LatentODEGenerativeModel, output_layers: nn.Module, output_loss_weight: float) -> None:
        if output_loss_weight <= 0.0:
            raise ValueError(f"output_loss_weight must be positive, got {output_loss_weight}")
        super().__init__()
        self.model: LatentODEGenerativeModel = model
        self.output_layers: nn.Module = output_layers
        self.output_loss_weight: float = output_loss_weight

    @override
    def forward(self, sequences: IrregularTensors) -> Tensor:
        outputs: Tensor = self.output_layers(self.model.infer(sequences).sample(self.training))
        return outputs

    @override
    def compute_training_loss(self, sequences: IrregularTensors, output_loss: Callable[[Tensor], Tensor]) -> Tensor:
        posterior: LatentPosterior = self.model.infer(sequences)
        latent: Tensor = posterior.sample(self.training)
        outputs: Tensor = self.output_layers(latent)
        return self.output_loss_weight * output_loss(outputs) + self.model.compute_negative_elbo(
            posterior, latent, sequences
        )
