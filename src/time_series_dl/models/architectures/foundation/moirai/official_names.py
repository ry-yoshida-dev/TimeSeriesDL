from ....components.pretrained import NameRule, ParameterNameMapping

_COMPONENT_PARAMETERS: tuple[tuple[str, str], ...] = (
    ("0.df", "student_t_degrees_of_freedom"),
    ("0.loc", "student_t_location"),
    ("0.scale", "student_t_scale"),
    ("1.loc", "normal_location"),
    ("2.total_count", "negative_binomial_total_count"),
    ("2.logits", "negative_binomial_logits"),
    ("3.loc", "log_normal_location"),
    ("3.scale", "log_normal_scale"),
)


class MoiraiOfficialNames:
    """
    Parameter names of the published Moirai checkpoints (``MoiraiModule`` of uni2ts) in terms of
    ``MoiraiForecastNetwork``.
    """

    @staticmethod
    def mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Returns
        -------
        ParameterNameMapping
            Renamings of the embeddings, the encoder layers and the distribution projections; nothing is ignored.
        """
        layer: str = r"^encoder\.layers\.(\d+)\."
        return ParameterNameMapping(
            rules=(
                NameRule(r"^mask_encoding\.", "model.mask_embedding."),
                NameRule(r"^in_proj\.", "model.input_projection."),
                NameRule(layer + r"self_attn\.q_proj\.", r"model.encoder.layers.\1.attention.query."),
                NameRule(layer + r"self_attn\.k_proj\.", r"model.encoder.layers.\1.attention.key."),
                NameRule(layer + r"self_attn\.v_proj\.", r"model.encoder.layers.\1.attention.value."),
                NameRule(layer + r"self_attn\.out_proj\.", r"model.encoder.layers.\1.attention.output."),
                NameRule(layer + r"self_attn\.q_norm\.", r"model.encoder.layers.\1.attention.query_norm."),
                NameRule(layer + r"self_attn\.k_norm\.", r"model.encoder.layers.\1.attention.key_norm."),
                NameRule(
                    layer + r"self_attn\.var_attn_bias\.emb\.", r"model.encoder.layers.\1.attention.variate_bias."
                ),
                NameRule(layer + r"ffn\.fc1\.", r"model.encoder.layers.\1.feed_forward.input."),
                NameRule(layer + r"ffn\.fc_gate\.", r"model.encoder.layers.\1.feed_forward.gate."),
                NameRule(layer + r"ffn\.fc2\.", r"model.encoder.layers.\1.feed_forward.output."),
                NameRule(layer + r"norm1\.", r"model.encoder.layers.\1.attention_norm."),
                NameRule(layer + r"norm2\.", r"model.encoder.layers.\1.feed_forward_norm."),
                NameRule(r"^encoder\.norm\.", "model.encoder.final_norm."),
                NameRule(r"^param_proj\.proj\.weights_logits\.", "model.distribution_head.mixture_logits."),
                *(
                    NameRule(
                        r"^param_proj\.proj\.components\." + published.replace(".", r"\.") + r"\.",
                        f"model.distribution_head.{name}.",
                    )
                    for published, name in _COMPONENT_PARAMETERS
                ),
            ),
        )
