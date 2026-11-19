from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .mixture import MoiraiMixture
from .output_projection import MoiraiOutputProjection


class MoiraiDistributionHead(nn.Module):
    """
    Projection of Moirai's token representations to the parameters of its mixture distribution for every value of
    the token's patch, with one multi-patch-size projection per parameter.

    Parameters
    ----------
    model_size : int
        Size of the token representations.
    patch_sizes : tuple[int, ...]
        The supported patch sizes.
    normal_scale : float
        Fixed standard deviation of the normal component.
    """

    def __init__(self, model_size: int, patch_sizes: tuple[int, ...], normal_scale: float) -> None:
        super().__init__()
        self.normal_scale: float = normal_scale
        self.mixture_logits: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 4)
        self.student_t_degrees_of_freedom: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.student_t_location: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.student_t_scale: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.normal_location: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.negative_binomial_total_count: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.negative_binomial_logits: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.log_normal_location: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)
        self.log_normal_scale: MoiraiOutputProjection = MoiraiOutputProjection(model_size, patch_sizes, 1)

    def forward(self, hidden: Tensor, patch_size: int) -> MoiraiMixture:
        """
        Compute the distribution of every value of every token's patch.

        Parameters
        ----------
        hidden : Tensor
            Token representations with shape (..., model_size).
        patch_size : int
            The patch size of the tokens.

        Returns
        -------
        MoiraiMixture
            The normalized distribution, with values of shape (..., max(patch_sizes)).
        """

        def project(projection: MoiraiOutputProjection) -> Tensor:
            return projection(hidden, patch_size).squeeze(-1)

        def positive(projection: MoiraiOutputProjection) -> Tensor:
            raw: Tensor = project(projection)
            return functional.softplus(raw).clamp_min(torch.finfo(raw.dtype).eps)

        return MoiraiMixture(
            mixture_logits=self.mixture_logits(hidden, patch_size),
            student_t_degrees_of_freedom=2.0 + functional.softplus(project(self.student_t_degrees_of_freedom)),
            student_t_location=project(self.student_t_location),
            student_t_scale=positive(self.student_t_scale),
            normal_location=project(self.normal_location),
            normal_scale=self.normal_scale,
            negative_binomial_total_count=functional.softplus(project(self.negative_binomial_total_count)),
            negative_binomial_logits=project(self.negative_binomial_logits),
            log_normal_location=project(self.log_normal_location),
            log_normal_scale=positive(self.log_normal_scale),
        )

    def __call__(self, hidden: Tensor, patch_size: int) -> MoiraiMixture:
        return cast(MoiraiMixture, super().__call__(hidden, patch_size))
