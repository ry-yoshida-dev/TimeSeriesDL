from enum import Enum

from torch import Tensor


class TemporalPooling(Enum):
    """
    Reduction of a sequence of feature vectors to one vector.

    Attributes
    ----------
    MEAN
        Mean over time.
    MAX
        Maximum over time.
    LAST
        The last time step, natural for causal encoders such as RNNs and TCNs.
    FLATTEN
        Every time step concatenated, keeping the position of each feature.
    """

    MEAN = "mean"
    MAX = "max"
    LAST = "last"
    FLATTEN = "flatten"

    def output_size(self, length: int, feature_size: int) -> int:
        """
        Get the size of the pooled vector.

        Parameters
        ----------
        length : int
            Number of time steps.
        feature_size : int
            Size of each feature vector.

        Returns
        -------
        int
            The size of the pooled vector.
        """
        match self:
            case TemporalPooling.MEAN | TemporalPooling.MAX | TemporalPooling.LAST:
                return feature_size
            case TemporalPooling.FLATTEN:
                return length * feature_size

    def apply(self, features: Tensor) -> Tensor:
        """
        Pool a batch of feature sequences.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, length, feature_size).

        Returns
        -------
        Tensor
            Pooled features with shape (batch, output_size).
        """
        match self:
            case TemporalPooling.MEAN:
                return features.mean(dim=1)
            case TemporalPooling.MAX:
                return features.amax(dim=1)
            case TemporalPooling.LAST:
                return features[:, -1, :]
            case TemporalPooling.FLATTEN:
                return features.flatten(start_dim=1)
