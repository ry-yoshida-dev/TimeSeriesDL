from dataclasses import dataclass

from ...interfaces.irregular_sequence_encoder import IrregularSequenceEncoder
from ...interfaces.irregular_summary_network import IrregularSummaryNetwork
from ...interfaces.shapes.irregular_summary import IrregularSummaryShape
from ..layers import Activation, MultilayerPerceptron
from .irregular_summary_network import EncodedIrregularSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class IrregularEncoderHead:
    """
    Fully connected output layers that turn an irregular sequence encoder into a summary network.

    Attributes
    ----------
    hidden_sizes : tuple[int, ...]
        Size of each hidden layer; a single linear layer when empty.
    activation : Activation
        Activation after each hidden layer.
    dropout : float
        Dropout probability after each hidden activation.

    Raises
    ------
    ValueError
        If a hidden size is not positive or dropout is not in [0, 1).
    """

    hidden_sizes: tuple[int, ...] = ()
    activation: Activation = Activation.RELU
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_output_layers(self, input_size: int, output_size: int) -> MultilayerPerceptron:
        """
        Build the untrained output layers.

        Parameters
        ----------
        input_size : int
            Size of the feature vectors.
        output_size : int
            Size of the output vectors.

        Returns
        -------
        MultilayerPerceptron
            The layers.
        """
        return MultilayerPerceptron(input_size, self.hidden_sizes, output_size, self.activation, self.dropout)

    def build_summary_network(
        self, encoder: IrregularSequenceEncoder, shape: IrregularSummaryShape
    ) -> IrregularSummaryNetwork:
        """
        Attach the output layers to an encoder.

        Parameters
        ----------
        encoder : IrregularSequenceEncoder
            The encoder.
        shape : IrregularSummaryShape
            The input and output dimensions.

        Returns
        -------
        IrregularSummaryNetwork
            The network.
        """
        return EncodedIrregularSummaryNetwork(encoder, self.build_output_layers(encoder.output_size, shape.output_size))
