from typing import override

from torch import Tensor

from ....interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....interfaces.irregular_tensors import IrregularTensors
from .generative_model import LatentODEGenerativeModel
from .latent_posterior import LatentPosterior


class LatentODEInterpolationNetwork(IrregularInterpolationNetwork):
    """
    Latent ODE interpolation and extrapolation: the posterior mean of the initial latent state is decoded at the
    query times. Training minimizes the negative evidence lower bound of the observed targets.

    Parameters
    ----------
    model : LatentODEGenerativeModel
        The generative model and its recognition network.
    """

    def __init__(self, model: LatentODEGenerativeModel) -> None:
        super().__init__()
        self.model: LatentODEGenerativeModel = model

    @override
    def forward(self, context: IrregularTensors, query_times: Tensor) -> Tensor:
        return self.model.decode(self.model.infer(context).mean, query_times)

    @override
    def compute_training_loss(self, context: IrregularTensors, targets: IrregularTensors) -> Tensor:
        posterior: LatentPosterior = self.model.infer(context)
        return self.model.compute_negative_elbo(posterior, posterior.sample(self.training), targets)
