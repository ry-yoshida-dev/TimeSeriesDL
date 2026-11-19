from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ....components.pretrained import NameRule, ParameterNameMapping

if TYPE_CHECKING:
    from .parameters import TTMParameters


class TTMOfficialNames:
    """
    Parameter names of the published TinyTimeMixer checkpoints (``TinyTimeMixerForPrediction`` of granite-tsfm) in
    terms of ``TTMForecastNetwork``.
    """

    @staticmethod
    def mapping(parameters: TTMParameters) -> ParameterNameMapping:
        """
        Get the translation of the published names.

        Parameters
        ----------
        parameters : TTMParameters
            The architecture, whose adaptive patching levels decide how the mixer layers are nested.

        Returns
        -------
        ParameterNameMapping
            Renamings of the patch embedding, both mixer stacks, the adapter and the head.
        """
        return ParameterNameMapping(
            rules=(
                NameRule(r"^backbone\.encoder\.patcher\.", "model.patch_embedding."),
                NameRule(r"^decoder\.adapter\.", "model.adapter."),
                NameRule(r"^head\.base_forecast_block\.", "model.head."),
                TTMOfficialNames._stack_rule(
                    "backbone.encoder.mlp_mixer_encoder.", "model.encoder.", parameters.adaptive_patching_levels
                ),
                TTMOfficialNames._stack_rule(
                    "decoder.decoder_block.", "model.decoder.", parameters.decoder_adaptive_patching_levels
                ),
                NameRule(r"\.norm\.norm\.", ".norm."),
                NameRule(r"\.mlp\.fc1\.", ".perceptron.hidden."),
                NameRule(r"\.mlp\.fc2\.", ".perceptron.output."),
                NameRule(r"\.gating_block\.attn_layer\.", ".gate.projection."),
            ),
        )

    @staticmethod
    def _stack_rule(published_prefix: str, prefix: str, adaptive_patching_levels: int) -> NameRule:
        start: str = "^" + re.escape(published_prefix)
        if adaptive_patching_levels > 0:
            return NameRule(start + r"mixers\.(\d+)\.mixer_layers\.(\d+)\.", prefix + r"blocks.\1.layers.\2.")
        return NameRule(start + r"mixers\.(\d+)\.", prefix + r"blocks.0.layers.\1.")
