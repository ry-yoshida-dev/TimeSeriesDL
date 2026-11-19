from enum import Enum

from torch import nn


class RecurrentCell(Enum):
    """
    Cell of a recurrent network.

    Attributes
    ----------
    LSTM
        Long short-term memory.
    GRU
        Gated recurrent unit.
    ELMAN
        Elman cell with tanh activation (the plain RNN).
    """

    LSTM = "lstm"
    GRU = "gru"
    ELMAN = "elman"

    def build(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float, is_bidirectional: bool
    ) -> nn.LSTM | nn.GRU | nn.RNN:
        """
        Build a batch-first recurrent layer stack of this cell.

        Parameters
        ----------
        input_size : int
            Size of the input vectors.
        hidden_size : int
            Size of the hidden state per direction.
        num_layers : int
            Number of stacked layers.
        dropout : float
            Dropout probability between layers.
        is_bidirectional : bool
            Whether each layer also reads the sequence backward.

        Returns
        -------
        nn.LSTM | nn.GRU | nn.RNN
            The recurrent layers.
        """
        between_layers: float = dropout if num_layers > 1 else 0.0
        match self:
            case RecurrentCell.LSTM:
                return nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=between_layers,
                    bidirectional=is_bidirectional,
                )
            case RecurrentCell.GRU:
                return nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=between_layers,
                    bidirectional=is_bidirectional,
                )
            case RecurrentCell.ELMAN:
                return nn.RNN(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=between_layers,
                    bidirectional=is_bidirectional,
                )
