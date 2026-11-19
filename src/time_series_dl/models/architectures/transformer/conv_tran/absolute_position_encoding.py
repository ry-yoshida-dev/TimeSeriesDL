from enum import Enum


class AbsolutePositionEncoding(Enum):
    """
    Absolute position encoding added to the tokens before attention.

    Attributes
    ----------
    TAPE
        Time absolute position encoding: sinusoids whose frequencies are scaled by ``model_size / length``, so the
        encodings of nearby positions stay distinguishable when the token vectors are small.
    SINUSOIDAL
        Fixed sinusoidal encoding of the original Transformer.
    LEARNABLE
        Learned vector per position.
    NONE
        No absolute position encoding.
    """

    TAPE = "tape"
    SINUSOIDAL = "sinusoidal"
    LEARNABLE = "learnable"
    NONE = "none"
