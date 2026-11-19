from ....components.pretrained import NameRule, ParameterNameMapping
from ....components.t5 import T5OfficialNames


class ChronosOfficialNames:
    """
    Parameter names of the published Chronos checkpoints (``T5ForConditionalGeneration`` of ``transformers``) in
    terms of ``ChronosForecastNetwork``.
    """

    @staticmethod
    def mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Returns
        -------
        ParameterNameMapping
            Renamings of the shared token embedding and both T5 stacks. The output layer is tied to the shared
            embedding in every published checkpoint, so the copies that some exports contain (``lm_head`` and the
            stacks' ``embed_tokens``) are ignored.
        """
        return ParameterNameMapping(
            rules=(
                NameRule(r"^shared\.", "model.token_embedding."),
                *T5OfficialNames.encoder_rules("encoder.", "model.encoder."),
                *T5OfficialNames.decoder_rules("decoder.", "model.decoder."),
            ),
            ignored=(r"(?:encoder|decoder)\.embed_tokens\.weight", r"lm_head\.weight"),
        )
