import math
from typing import Final, cast

import torch
from torch import Tensor, nn

_MASKED_SCORE: Final[float] = -1e9


class MultiTimeAttention(nn.Module):
    """
    Multi-time attention (Shukla and Marlin, 2021): each query time attends to the key times through their learned
    time embeddings, separately per value channel so that only the times where a channel is observed take part.

    Parameters
    ----------
    value_size : int
        Number of value channels.
    output_size : int
        Size of the output vector of each query time.
    embedding_size : int
        Size of the time embeddings.
    num_heads : int
        Number of attention heads; each projects the embeddings to ``embedding_size // num_heads``.

    Raises
    ------
    ValueError
        If a size is not positive or embedding_size is not divisible by num_heads.
    """

    def __init__(self, value_size: int, output_size: int, embedding_size: int, num_heads: int) -> None:
        for name, value in (
            ("value_size", value_size),
            ("output_size", output_size),
            ("embedding_size", embedding_size),
            ("num_heads", num_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if embedding_size % num_heads != 0:
            raise ValueError(f"embedding_size {embedding_size} must be divisible by num_heads {num_heads}")
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = embedding_size // num_heads
        self.query_projection: nn.Linear = nn.Linear(embedding_size, embedding_size)
        self.key_projection: nn.Linear = nn.Linear(embedding_size, embedding_size)
        self.output_projection: nn.Linear = nn.Linear(num_heads * value_size, output_size)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Attend from the query times to the key times.

        Parameters
        ----------
        queries : Tensor
            Embeddings of the query times with shape (batch, num_queries, embedding_size); a batch of 1 is shared.
        keys : Tensor
            Embeddings of the key times with shape (batch, num_keys, embedding_size).
        values : Tensor
            Values at the key times with shape (batch, num_keys, value_size).
        mask : Tensor | None
            1.0 where a value takes part and 0.0 elsewhere, with the shape of ``values``; every value takes part
            when None. A channel with no value taking part yields zero.

        Returns
        -------
        Tensor
            Outputs with shape (batch, num_queries, output_size).
        """
        batch_size: int = values.shape[0]
        num_queries: int = queries.shape[1]
        projected_queries: Tensor = self._split_heads(self.query_projection(queries))
        projected_keys: Tensor = self._split_heads(self.key_projection(keys))
        scores: Tensor = projected_queries @ projected_keys.transpose(-2, -1) / math.sqrt(self.head_size)
        attended: Tensor
        if mask is None:
            attended = torch.softmax(scores, dim=-1) @ values.unsqueeze(1)
        else:
            # (batch, heads, queries, keys, channels): a softmax over the keys of each channel.
            channel_mask: Tensor = mask.unsqueeze(1).unsqueeze(2)
            channel_scores: Tensor = torch.where(
                channel_mask > 0, scores.unsqueeze(-1), torch.full_like(channel_mask, _MASKED_SCORE)
            )
            exponentials: Tensor = torch.exp(channel_scores - channel_scores.amax(dim=3, keepdim=True)) * channel_mask
            weights: Tensor = exponentials / exponentials.sum(dim=3, keepdim=True).clamp(
                min=torch.finfo(values.dtype).tiny
            )
            attended = (weights * values.unsqueeze(1).unsqueeze(2)).sum(dim=3)
        combined: Tensor = attended.transpose(1, 2).reshape(batch_size, num_queries, -1)
        outputs: Tensor = self.output_projection(combined)
        return outputs

    def _split_heads(self, embeddings: Tensor) -> Tensor:
        return embeddings.view(embeddings.shape[0], embeddings.shape[1], self.num_heads, self.head_size).transpose(1, 2)

    def __call__(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> Tensor:
        return cast(Tensor, super().__call__(queries, keys, values, mask))
