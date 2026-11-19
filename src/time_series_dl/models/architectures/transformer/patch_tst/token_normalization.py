from enum import Enum

from torch import nn


class TokenNormalization(Enum):
    """
    Normalization of the patch tokens inside the Transformer encoder.

    Attributes
    ----------
    BATCH
        Batch normalization of each feature over the batch and the patches, which the PatchTST paper prefers.
    LAYER
        Layer normalization of each token, as in the original Transformer.
    """

    BATCH = "batch"
    LAYER = "layer"

    def build(self, model_size: int) -> nn.BatchNorm1d | nn.LayerNorm:
        """
        Build the normalization of tokens flattened to shape (tokens, model_size).

        Parameters
        ----------
        model_size : int
            Size of each token vector.

        Returns
        -------
        nn.BatchNorm1d | nn.LayerNorm
            The normalization.
        """
        match self:
            case TokenNormalization.BATCH:
                return nn.BatchNorm1d(model_size)
            case TokenNormalization.LAYER:
                return nn.LayerNorm(model_size)
