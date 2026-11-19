from typing import override

import torch
from torch import Tensor, nn

from ....interfaces.irregular_sequence_encoder import IrregularSequenceEncoder
from ....interfaces.irregular_tensors import IrregularTensors
from .multi_time_attention import MultiTimeAttention
from .time_embedding import LearnedTimeEmbedding


class MTANSequenceEncoder(IrregularSequenceEncoder):
    """
    mTAND encoder of the classifier (mTAND-Enc): multi-time attention re-represents the observed values and masks at
    fixed reference times in [0, 1], and a GRU reads the reference times; its last state is the feature vector.

    Parameters
    ----------
    input_channels : int
        Number of channels of the sequences.
    hidden_size : int
        Size of the attention outputs and of the GRU state.
    time_embedding_size : int
        Size of the learned time embeddings.
    num_heads : int
        Number of attention heads.
    num_reference_points : int
        Number of equally spaced reference times in [0, 1].

    Raises
    ------
    ValueError
        If a size is not positive or the embedding size is not divisible by the number of heads.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        time_embedding_size: int,
        num_heads: int,
        num_reference_points: int,
    ) -> None:
        if num_reference_points <= 0:
            raise ValueError(f"num_reference_points must be positive, got {num_reference_points}")
        super().__init__()
        self.hidden_size: int = hidden_size
        self.reference_times: Tensor
        self.register_buffer("reference_times", torch.linspace(0.0, 1.0, num_reference_points))
        self.time_embedding: LearnedTimeEmbedding = LearnedTimeEmbedding(time_embedding_size)
        self.attention: MultiTimeAttention = MultiTimeAttention(
            2 * input_channels, hidden_size, time_embedding_size, num_heads
        )
        self.recurrence: nn.GRU = nn.GRU(hidden_size, hidden_size, batch_first=True)

    @property
    @override
    def output_size(self) -> int:
        return self.hidden_size

    @override
    def forward(self, sequences: IrregularTensors) -> Tensor:
        mask: Tensor = sequences.observed_mask
        attended: Tensor = self.attention(
            self.time_embedding(self.reference_times).unsqueeze(0),
            self.time_embedding(sequences.times),
            torch.cat([sequences.values, mask], dim=-1),
            torch.cat([mask, mask], dim=-1),
        )
        final_state: Tensor = self.recurrence(attended)[1]
        return final_state[-1]
