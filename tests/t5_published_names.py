"""Parameter names of published T5 stacks, written out independently of the mappings under test."""

from time_series_dl.models.components.t5 import T5FeedForwardKind


def published_stack_names(prefix: str, num_layers: int, is_decoder: bool, kind: T5FeedForwardKind) -> list[str]:
    """Parameter names of a T5 stack in Hugging Face ``transformers``."""
    feed_forward: list[str] = ["wi", "wo"] if kind == T5FeedForwardKind.RELU else ["wi_0", "wi_1", "wo"]
    names: list[str] = [f"{prefix}block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    for block in range(num_layers):
        layer: str = f"{prefix}block.{block}.layer."
        names += [f"{layer}0.SelfAttention.{projection}.weight" for projection in "qkvo"]
        names.append(f"{layer}0.layer_norm.weight")
        last: int = 1
        if is_decoder:
            names += [f"{layer}1.EncDecAttention.{projection}.weight" for projection in "qkvo"]
            names.append(f"{layer}1.layer_norm.weight")
            last = 2
        names += [f"{layer}{last}.DenseReluDense.{weight}.weight" for weight in feed_forward]
        names.append(f"{layer}{last}.layer_norm.weight")
    names.append(f"{prefix}final_layer_norm.weight")
    return names
