from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.distributions import Categorical, Gamma, LogNormal, Normal, StudentT


@dataclass(frozen=True, eq=False)
class MoiraiMixture:
    """
    Moirai's predictive distribution of normalized values: a mixture of a Student's t, a normal with a small fixed
    scale, a negative binomial and a log-normal distribution, with weights and parameters per value.

    Every tensor has the shape of the values, except ``mixture_logits``, which has one more trailing dimension of 4
    (the components in the order above).

    Attributes
    ----------
    mixture_logits : Tensor
        Unnormalized log weights of the components.
    student_t_degrees_of_freedom : Tensor
        Degrees of freedom of the Student's t component (above 2).
    student_t_location : Tensor
        Location of the Student's t component.
    student_t_scale : Tensor
        Positive scale of the Student's t component.
    normal_location : Tensor
        Mean of the normal component.
    normal_scale : float
        Fixed standard deviation of the normal component.
    negative_binomial_total_count : Tensor
        Non-negative total count of the negative binomial component.
    negative_binomial_logits : Tensor
        Log-odds of the negative binomial component.
    log_normal_location : Tensor
        Mean of the logarithm of the log-normal component.
    log_normal_scale : Tensor
        Positive standard deviation of the logarithm of the log-normal component.
    """

    mixture_logits: Tensor
    student_t_degrees_of_freedom: Tensor
    student_t_location: Tensor
    student_t_scale: Tensor
    normal_location: Tensor
    normal_scale: float
    negative_binomial_total_count: Tensor
    negative_binomial_logits: Tensor
    log_normal_location: Tensor
    log_normal_scale: Tensor

    def log_prob(self, values: Tensor) -> Tensor:
        """
        Compute the log density of normalized values.

        Values outside the support of a component (negative for the negative binomial, non-positive for the
        log-normal) get no density from it, and its weight is then dropped as in the original implementation.

        Parameters
        ----------
        values : Tensor
            Normalized values with the shape of the parameters.

        Returns
        -------
        Tensor
            The log densities, with the same shape.
        """
        is_non_negative: Tensor = values >= 0
        is_positive: Tensor = values > 0
        # Unsupported values are replaced by 1, which lies in every support, and their densities are discarded.
        counts: Tensor = torch.where(is_non_negative, values, 1.0)
        positives: Tensor = torch.where(is_positive, values, 1.0)
        negative_infinity: Tensor = torch.full_like(values, float("-inf"))
        components: Tensor = torch.stack(
            [
                StudentT(
                    self.student_t_degrees_of_freedom,
                    self.student_t_location,
                    self.student_t_scale,
                    validate_args=False,
                ).log_prob(values),
                Normal(self.normal_location, self.normal_scale, validate_args=False).log_prob(values),
                torch.where(is_non_negative, self._negative_binomial_log_prob(counts), negative_infinity),
                torch.where(
                    is_positive,
                    LogNormal(self.log_normal_location, self.log_normal_scale, validate_args=False).log_prob(positives),
                    negative_infinity,
                ),
            ],
            dim=-1,
        )
        weights: Tensor = torch.log_softmax(self.mixture_logits, dim=-1)
        weights = torch.where(torch.isinf(components), 0.0, weights)
        return torch.logsumexp(weights + components, dim=-1)

    def sample(self, num_samples: int) -> Tensor:
        """
        Draw normalized samples with PyTorch's global random number generator.

        Parameters
        ----------
        num_samples : int
            Number of samples per value.

        Returns
        -------
        Tensor
            The samples with shape (num_samples, *shape of the values).
        """
        size: torch.Size = torch.Size((num_samples,))
        with torch.no_grad():
            student_t: Tensor = StudentT(
                self.student_t_degrees_of_freedom, self.student_t_location, self.student_t_scale, validate_args=False
            ).sample(size)
            normal: Tensor = Normal(self.normal_location, self.normal_scale, validate_args=False).sample(size)
            rate: Tensor = torch.exp(-self.negative_binomial_logits)
            negative_binomial: Tensor = torch.poisson(
                Gamma(self.negative_binomial_total_count, rate, validate_args=False).sample(size)
            )
            log_normal: Tensor = (
                Normal(self.log_normal_location, self.log_normal_scale, validate_args=False).sample(size).exp()
            )
            choices: Tensor = Categorical(logits=self.mixture_logits, validate_args=False).sample(size)
            stacked: Tensor = torch.stack([student_t, normal, negative_binomial, log_normal], dim=-1)
            return stacked.gather(-1, choices.unsqueeze(-1)).squeeze(-1)

    def _negative_binomial_log_prob(self, values: Tensor) -> Tensor:
        total_count: Tensor = self.negative_binomial_total_count
        logits: Tensor = self.negative_binomial_logits
        unnormalized: Tensor = total_count * functional.logsigmoid(-logits) + functional.logsigmoid(logits) * values
        log_beta: Tensor = torch.lgamma(1 + values) + torch.lgamma(total_count) - torch.lgamma(1 + values + total_count)
        return unnormalized - (log_beta + torch.log(total_count + values))
