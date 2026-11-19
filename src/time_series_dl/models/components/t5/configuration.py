from dataclasses import dataclass

from .feed_forward_kind import T5FeedForwardKind


@dataclass(frozen=True, kw_only=True)
class T5Configuration:
    """
    Sizes of a T5 encoder or decoder stack (Raffel et al., 2020), named after the fields of the published configs.

    Attributes
    ----------
    model_size : int
        Size of the hidden states (``d_model``).
    key_value_size : int
        Size of each attention head (``d_kv``); the heads together need not equal model_size.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks (``d_ff``).
    num_layers : int
        Number of blocks.
    feed_forward_kind : T5FeedForwardKind
        ReLU or gated-GELU feed-forward networks.
    relative_attention_num_buckets : int
        Number of buckets of relative positions.
    relative_attention_max_distance : int
        Distance beyond which relative positions share a bucket.
    dropout : float
        Dropout probability of the embeddings, the attention weights and every residual branch.
    layer_norm_epsilon : float
        Added to the mean square in the root-mean-square layer normalization.

    Raises
    ------
    ValueError
        If a size is not positive, the buckets are too few for their distance, or dropout is not in [0, 1).
    """

    model_size: int
    key_value_size: int
    num_heads: int
    feedforward_size: int
    num_layers: int
    feed_forward_kind: T5FeedForwardKind = T5FeedForwardKind.RELU
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("key_value_size", self.key_value_size),
            ("num_heads", self.num_heads),
            ("feedforward_size", self.feedforward_size),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.relative_attention_num_buckets < 4:
            raise ValueError(
                f"relative_attention_num_buckets must be at least 4, got {self.relative_attention_num_buckets}"
            )
        if self.relative_attention_max_distance <= self.relative_attention_num_buckets // 4:
            raise ValueError(
                "relative_attention_max_distance must exceed a quarter of the buckets, got "
                + f"{self.relative_attention_max_distance}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError(f"layer_norm_epsilon must be positive, got {self.layer_norm_epsilon}")

    @property
    def inner_size(self) -> int:
        """
        Get the size of all attention heads together.

        Returns
        -------
        int
            ``num_heads * key_value_size``.
        """
        return self.num_heads * self.key_value_size
