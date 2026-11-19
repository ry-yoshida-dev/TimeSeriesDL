"""Structural type of the parameters of decomposition Transformers."""

from typing import Protocol

from ....components.attention.attention_mechanism import AttentionMechanism
from ....components.layers.activation import Activation


class DecompositionTransformerConfiguration(Protocol):
    """
    Sizes of a decomposition Transformer and the attention mechanisms that distinguish its variants.
    """

    @property
    def model_size(self) -> int:
        """
        Get the size of each token vector.

        Returns
        -------
        int
            The model size.
        """
        ...

    @property
    def num_heads(self) -> int:
        """
        Get the number of attention heads.

        Returns
        -------
        int
            The number of heads, which divides the model size.
        """
        ...

    @property
    def num_encoder_layers(self) -> int:
        """
        Get the number of encoder layers.

        Returns
        -------
        int
            The number of encoder layers.
        """
        ...

    @property
    def num_decoder_layers(self) -> int:
        """
        Get the number of decoder layers.

        Returns
        -------
        int
            The number of decoder layers.
        """
        ...

    @property
    def feedforward_size(self) -> int:
        """
        Get the hidden size of the position-wise feed-forward networks.

        Returns
        -------
        int
            The hidden size.
        """
        ...

    @property
    def moving_average_size(self) -> int:
        """
        Get the length of the moving average of the series decompositions.

        Returns
        -------
        int
            The odd window length.
        """
        ...

    @property
    def dropout(self) -> float:
        """
        Get the dropout probability of the embeddings and sublayer outputs.

        Returns
        -------
        float
            The probability in [0, 1).
        """
        ...

    @property
    def activation(self) -> Activation:
        """
        Get the activation of the feed-forward networks.

        Returns
        -------
        Activation
            The activation.
        """
        ...

    @property
    def label_length(self) -> int:
        """
        Get the number of last past steps that start the decoder input.

        Returns
        -------
        int
            The label length.
        """
        ...

    def build_self_attention(self, length: int) -> AttentionMechanism:
        """
        Build the self-attention mechanism of a sequence.

        Parameters
        ----------
        length : int
            Number of time steps of the sequence.

        Returns
        -------
        AttentionMechanism
            The mechanism.
        """
        ...

    def build_cross_attention(self, query_length: int, key_length: int) -> AttentionMechanism:
        """
        Build the cross-attention mechanism from the decoder to the encoder output.

        Parameters
        ----------
        query_length : int
            Number of time steps of the decoder.
        key_length : int
            Number of time steps of the encoder output.

        Returns
        -------
        AttentionMechanism
            The mechanism.
        """
        ...
