import re

from ..pretrained import NameRule


class T5OfficialNames:
    """
    Renamings of the parameter names of published T5 stacks (the naming of Hugging Face ``transformers``, such as
    ``encoder.block.0.layer.0.SelfAttention.q.weight``) into the names of ``T5Encoder`` and ``T5Decoder``.
    """

    @staticmethod
    def encoder_rules(published_prefix: str, prefix: str) -> tuple[NameRule, ...]:
        """
        Get the renamings of an encoder stack.

        Parameters
        ----------
        published_prefix : str
            Prefix of the stack in the published names, such as ``"encoder."``.
        prefix : str
            Prefix of the ``T5Encoder`` in the network's state dict, such as ``"encoder."``.

        Returns
        -------
        tuple[NameRule, ...]
            The rules, to be applied in order.
        """
        block: str = re.escape(published_prefix) + r"block\.(\d+)\.layer\."
        return (
            *T5OfficialNames._shared_rules(published_prefix, prefix),
            NameRule(block + r"1\.DenseReluDense\.", prefix + r"blocks.\1.feed_forward."),
            NameRule(block + r"1\.layer_norm\.", prefix + r"blocks.\1.feed_forward_norm."),
            *T5OfficialNames._leaf_rules(prefix),
        )

    @staticmethod
    def decoder_rules(published_prefix: str, prefix: str) -> tuple[NameRule, ...]:
        """
        Get the renamings of a decoder stack.

        Parameters
        ----------
        published_prefix : str
            Prefix of the stack in the published names, such as ``"decoder."``.
        prefix : str
            Prefix of the ``T5Decoder`` in the network's state dict, such as ``"decoder."``.

        Returns
        -------
        tuple[NameRule, ...]
            The rules, to be applied in order.
        """
        block: str = re.escape(published_prefix) + r"block\.(\d+)\.layer\."
        return (
            *T5OfficialNames._shared_rules(published_prefix, prefix),
            NameRule(block + r"1\.EncDecAttention\.", prefix + r"blocks.\1.cross_attention."),
            NameRule(block + r"1\.layer_norm\.", prefix + r"blocks.\1.cross_attention_norm."),
            NameRule(block + r"2\.DenseReluDense\.", prefix + r"blocks.\1.feed_forward."),
            NameRule(block + r"2\.layer_norm\.", prefix + r"blocks.\1.feed_forward_norm."),
            *T5OfficialNames._leaf_rules(prefix),
        )

    @staticmethod
    def _shared_rules(published_prefix: str, prefix: str) -> tuple[NameRule, ...]:
        start: str = "^" + re.escape(published_prefix)
        return (
            NameRule(
                start + r"block\.0\.layer\.0\.SelfAttention\.relative_attention_bias\.",
                prefix + "position_bias.embedding.",
            ),
            NameRule(start + r"block\.(\d+)\.layer\.0\.SelfAttention\.", prefix + r"blocks.\1.self_attention."),
            NameRule(start + r"block\.(\d+)\.layer\.0\.layer_norm\.", prefix + r"blocks.\1.self_attention_norm."),
            NameRule(start + r"final_layer_norm\.", prefix + "final_norm."),
        )

    @staticmethod
    def _leaf_rules(prefix: str) -> tuple[NameRule, ...]:
        attention: str = "^(" + re.escape(prefix) + r"blocks\.\d+\.(?:self|cross)_attention)\."
        feed_forward: str = "^(" + re.escape(prefix) + r"blocks\.\d+\.feed_forward)\."
        return (
            NameRule(attention + r"q\.weight$", r"\1.query.weight"),
            NameRule(attention + r"k\.weight$", r"\1.key.weight"),
            NameRule(attention + r"v\.weight$", r"\1.value.weight"),
            NameRule(attention + r"o\.weight$", r"\1.output.weight"),
            NameRule(feed_forward + r"wi\.weight$", r"\1.input.weight"),
            NameRule(feed_forward + r"wi_0\.weight$", r"\1.gate.weight"),
            NameRule(feed_forward + r"wi_1\.weight$", r"\1.input.weight"),
            NameRule(feed_forward + r"wo\.weight$", r"\1.output.weight"),
        )
