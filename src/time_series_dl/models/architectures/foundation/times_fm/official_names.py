from ....components.pretrained import NameRule, ParameterNameMapping


class TimesFMOfficialNames:
    """
    Parameter names of the published TimesFM PyTorch checkpoints (``TimesFmModelForPrediction`` of Hugging Face
    ``transformers``, whose ``model.safetensors`` the repository holds) in terms of ``TimesFMForecastNetwork``.
    """

    @staticmethod
    def mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Returns
        -------
        ParameterNameMapping
            Renamings of the residual blocks, the frequency embedding and the decoder layers; nothing is ignored.
        """
        layer: str = r"^decoder\.layers\.(\d+)\."
        return ParameterNameMapping(
            rules=(
                NameRule(r"^decoder\.input_ff_layer\.", "model.input_embedding."),
                NameRule(r"^horizon_ff_layer\.", "model.output_projection."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.input_layer\.", r"\1.hidden."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.output_layer\.", r"\1.output."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.residual_layer\.", r"\1.residual."),
                NameRule(r"^decoder\.freq_emb\.", "model.frequency_embedding."),
                NameRule(layer + r"input_layernorm\.", r"model.layers.\1.attention_norm."),
                NameRule(layer + r"self_attn\.q_proj\.", r"model.layers.\1.attention.query."),
                NameRule(layer + r"self_attn\.k_proj\.", r"model.layers.\1.attention.key."),
                NameRule(layer + r"self_attn\.v_proj\.", r"model.layers.\1.attention.value."),
                NameRule(layer + r"self_attn\.o_proj\.", r"model.layers.\1.attention.output."),
                NameRule(layer + r"self_attn\.scaling$", r"model.layers.\1.attention.query_scale"),
                NameRule(layer + r"mlp\.layer_norm\.", r"model.layers.\1.feed_forward.norm."),
                NameRule(layer + r"mlp\.gate_proj\.", r"model.layers.\1.feed_forward.hidden."),
                NameRule(layer + r"mlp\.down_proj\.", r"model.layers.\1.feed_forward.output."),
            ),
        )
