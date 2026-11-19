from typing import override

from torch import Tensor

from ....interfaces.irregular_sequence_encoder import IrregularSequenceEncoder
from ....interfaces.irregular_tensors import IrregularTensors
from .recurrence import ODERNNRecurrence


class ODERNNSequenceEncoder(IrregularSequenceEncoder):
    """
    ODE-RNN read forward from time 0; the hidden state at each sample's last observation is the feature vector.

    Parameters
    ----------
    recurrence : ODERNNRecurrence
        The ODE-RNN.
    """

    def __init__(self, recurrence: ODERNNRecurrence) -> None:
        super().__init__()
        self.recurrence: ODERNNRecurrence = recurrence

    @property
    @override
    def output_size(self) -> int:
        return self.recurrence.hidden_size

    @override
    def forward(self, sequences: IrregularTensors) -> Tensor:
        return self.recurrence(sequences, is_backward=False)
