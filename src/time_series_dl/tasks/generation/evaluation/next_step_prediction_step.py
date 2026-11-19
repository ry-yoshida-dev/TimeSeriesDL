from dataclasses import dataclass

from torch import Tensor

from .recurrent_scorer import RecurrentScorer


@dataclass(frozen=True, eq=False)
class NextStepPredictionStep:
    """
    Mean absolute error of a post-hoc predictor of the next step's target at every step.

    Attributes
    ----------
    network : RecurrentScorer
        The predictor, with one output per step.
    """

    network: RecurrentScorer

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The inputs with shape (batch, length - 1, input_size) and the next steps' targets with shape
            (batch, length - 1, 1).
        is_training : bool
            Unused; the loss is the same while training and validating.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return (self.network(batch[0]) - batch[1]).abs().mean()
