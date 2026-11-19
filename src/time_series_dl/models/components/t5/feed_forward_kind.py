from enum import Enum


class T5FeedForwardKind(Enum):
    """
    Feed-forward network of the T5 blocks.

    Attributes
    ----------
    RELU
        One ReLU hidden layer, as in the original T5.
    GATED_GELU
        A tanh-approximated GELU gate multiplied with a linear projection, as in T5 v1.1 and Flan-T5.
    """

    RELU = "relu"
    GATED_GELU = "gated_gelu"
