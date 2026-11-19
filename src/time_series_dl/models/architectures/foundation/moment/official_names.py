from typing import Final

from ....components.pretrained import NameRule, ParameterNameMapping
from ....components.t5 import T5OfficialNames

# The unused token embedding of the T5 backbone, and the fixed sinusoidal position table that the networks compute
# themselves.
_ALWAYS_IGNORED: Final[tuple[str, ...]] = (
    r"encoder\.embed_tokens\.weight",
    r"patch_embedding\.position_embedding\.pe",
)


class MOMENTOfficialNames:
    """
    Parameter names of the published MOMENT checkpoints (``MOMENTPipeline`` of momentfm) in terms of the MOMENT
    modules.
    """

    @staticmethod
    def reconstruction_mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names into a reconstruction network: the encoder and the pretraining
        reconstruction head.

        Returns
        -------
        ParameterNameMapping
            The mapping.
        """
        return MOMENTOfficialNames._mapping("encoder.", _ALWAYS_IGNORED)

    @staticmethod
    def encoder_mapping() -> ParameterNameMapping:
        """
        Get the translation of the published names into a ``MOMENTEncoder``, for networks with a new head; the
        pretraining head is ignored.

        Returns
        -------
        ParameterNameMapping
            The mapping.
        """
        return MOMENTOfficialNames._mapping("", (*_ALWAYS_IGNORED, r"head\.linear\.(?:weight|bias)"))

    @staticmethod
    def _mapping(prefix: str, ignored: tuple[str, ...]) -> ParameterNameMapping:
        return ParameterNameMapping(
            rules=(
                *T5OfficialNames.encoder_rules("encoder.", prefix + "transformer."),
                NameRule(r"^patch_embedding\.", prefix + "patch_embedding."),
            ),
            ignored=ignored,
        )
