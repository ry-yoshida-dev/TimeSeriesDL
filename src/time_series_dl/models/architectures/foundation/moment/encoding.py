from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class MOMENTEncoding:
    """
    Encoder states of a batch of univariate series and the statistics their values were normalized with.

    Attributes
    ----------
    states : Tensor
        Patch states with shape (batch, num_patches, model_size).
    location : Tensor
        Mean of the observed values of each series with shape (batch, 1).
    scale : Tensor
        Standard deviation (plus epsilon) of the observed values of each series with shape (batch, 1).
    """

    states: Tensor
    location: Tensor
    scale: Tensor

    def denormalize(self, values: Tensor) -> Tensor:
        """
        Map normalized values back to the scale of the series.

        Parameters
        ----------
        values : Tensor
            Normalized values with shape (batch, ...).

        Returns
        -------
        Tensor
            The values in the scale of the series.
        """
        shape: tuple[int, ...] = (values.shape[0],) + (1,) * (values.dim() - 1)
        return values * self.scale.reshape(shape) + self.location.reshape(shape)
