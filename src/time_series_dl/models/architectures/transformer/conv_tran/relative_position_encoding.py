from enum import Enum


class RelativePositionEncoding(Enum):
    """
    Relative position encoding inside the attention.

    Attributes
    ----------
    ERPE
        Efficient relative position encoding: a learned scalar per head and relative distance added to the attention
        weights after the softmax.
    NONE
        Plain scaled dot-product attention.
    """

    ERPE = "erpe"
    NONE = "none"
