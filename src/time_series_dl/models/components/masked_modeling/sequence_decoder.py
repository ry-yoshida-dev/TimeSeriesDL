from typing import cast

from torch import Tensor, nn


class SequenceDecoder(nn.Module):
    """
    Linear map from a sequence of feature vectors back to the values of a sequence, as masked modeling reconstructs.

    When the features keep the length of the values, each step's features are mapped to its values; otherwise all
    flattened features are mapped to the whole sequence.

    Parameters
    ----------
    feature_length : int
        Number of feature vectors per sequence.
    feature_size : int
        Size of each feature vector.
    length : int
        Number of time steps of the values.
    channels : int
        Number of channels of the values.

    Raises
    ------
    ValueError
        If a dimension is not positive.
    """

    def __init__(self, feature_length: int, feature_size: int, length: int, channels: int) -> None:
        if min(feature_length, feature_size, length, channels) <= 0:
            raise ValueError(
                "dimensions must be positive, got " + f"{feature_length}, {feature_size}, {length} and {channels}"
            )
        super().__init__()
        self.length: int = length
        self.channels: int = channels
        self.is_stepwise: bool = feature_length == length
        self.projection: nn.Linear = (
            nn.Linear(feature_size, channels)
            if self.is_stepwise
            else nn.Linear(feature_length * feature_size, length * channels)
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Decode a batch of feature sequences.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, feature_length, feature_size).

        Returns
        -------
        Tensor
            Values with shape (batch, length, channels).
        """
        if self.is_stepwise:
            stepwise: Tensor = self.projection(features)
            return stepwise
        flattened: Tensor = self.projection(features.flatten(start_dim=1))
        return flattened.reshape(-1, self.length, self.channels)

    def __call__(self, features: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features))
