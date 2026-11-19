from ....components.pretrained import NameRule, ParameterNameMapping
from ....components.t5 import T5OfficialNames


class ChronosBoltOfficialNames:
    """
    Parameter names of the published Chronos-Bolt checkpoints (``ChronosBoltModelForForecasting`` of
    chronos-forecasting) in terms of ``ChronosBoltForecastNetwork``.
    """

    @staticmethod
    def mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Returns
        -------
        ParameterNameMapping
            Renamings of the embeddings, the residual blocks and both T5 stacks; the copies of the shared embedding
            that some exports of tied T5 weights contain are ignored.
        """
        return ParameterNameMapping(
            rules=(
                NameRule(r"^shared\.", "model.token_embedding."),
                NameRule(r"^input_patch_embedding\.", "model.input_embedding."),
                NameRule(r"^output_patch_embedding\.", "model.output_projection."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.hidden_layer\.", r"\1.hidden."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.output_layer\.", r"\1.output."),
                NameRule(r"^(model\.(?:input_embedding|output_projection))\.residual_layer\.", r"\1.residual."),
                *T5OfficialNames.encoder_rules("encoder.", "model.encoder."),
                *T5OfficialNames.decoder_rules("decoder.", "model.decoder."),
            ),
            ignored=(r"(?:encoder|decoder)\.embed_tokens\.weight", r"lm_head\.weight"),
        )
