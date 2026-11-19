from ....components.pretrained import NameRule, ParameterNameMapping


class TimerOfficialNames:
    """
    Parameter names of the published Timer checkpoints (``TimerForPrediction`` of the repository's
    ``modeling_timer.py``) in terms of ``TimerForecastNetwork``.
    """

    @staticmethod
    def mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Returns
        -------
        ParameterNameMapping
            Renamings of the patch embedding, the decoder layers, the final normalization and the single output head.
        """
        layer: str = r"^model\.layers\.(\d+)\."
        return ParameterNameMapping(
            rules=(
                NameRule(r"^model\.embed_layer\.emb\.", "model.patch_embedding."),
                NameRule(r"^model\.norm\.", "model.final_norm."),
                NameRule(r"^lm_heads\.0\.", "model.head."),
                NameRule(layer + r"self_attn\.q_proj\.", r"model.layers.\1.attention.query."),
                NameRule(layer + r"self_attn\.k_proj\.", r"model.layers.\1.attention.key."),
                NameRule(layer + r"self_attn\.v_proj\.", r"model.layers.\1.attention.value."),
                NameRule(layer + r"self_attn\.o_proj\.", r"model.layers.\1.attention.output."),
                NameRule(layer + r"ffn_layer\.gate_proj\.", r"model.layers.\1.feed_forward.gate."),
                NameRule(layer + r"ffn_layer\.up_proj\.", r"model.layers.\1.feed_forward.input."),
                NameRule(layer + r"ffn_layer\.down_proj\.", r"model.layers.\1.feed_forward.output."),
                NameRule(layer + r"norm1\.", r"model.layers.\1.attention_norm."),
                NameRule(layer + r"norm2\.", r"model.layers.\1.feed_forward_norm."),
            ),
        )
