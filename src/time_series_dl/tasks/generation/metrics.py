from __future__ import annotations

from dataclasses import dataclass

from ...array_types import FloatArray
from .evaluation import (
    AutocorrelationDistance,
    DiscriminativeScore,
    GenerationEvaluationSettings,
    MarginalDistance,
    PredictiveScore,
)


@dataclass(frozen=True)
class GenerationMetrics:
    """
    Fidelity and usefulness of synthetic sequences compared with real ones; lower is better for every metric.

    Attributes
    ----------
    discriminative_score : float
        ``|accuracy - 0.5|`` of a post-hoc classifier of real versus synthetic sequences, in [0, 0.5].
    predictive_score : float
        Mean absolute error on real sequences of a next-step predictor trained on synthetic ones.
    marginal_distance : float
        Quantile distance between the per-channel value distributions.
    autocorrelation_distance : float
        Mean absolute difference of the mean autocorrelation functions, in [0, 2].
    """

    discriminative_score: float
    predictive_score: float
    marginal_distance: float
    autocorrelation_distance: float

    @classmethod
    def compute(
        cls, real: FloatArray, synthetic: FloatArray, settings: GenerationEvaluationSettings
    ) -> GenerationMetrics:
        """
        Compute every metric.

        Parameters
        ----------
        real : FloatArray
            Real sequences with shape (num_real, length, channels), at least two sequences of two steps.
        synthetic : FloatArray
            Synthetic sequences with the same length and channels, at least two.
        settings : GenerationEvaluationSettings
            The post-hoc training and the parameters of the distances.

        Returns
        -------
        GenerationMetrics
            The metrics.

        Raises
        ------
        ValueError
            If the shapes differ, or there are too few sequences or steps.
        """
        return cls(
            discriminative_score=DiscriminativeScore(
                settings.discriminator_training, settings.test_fraction, settings.split_seed
            ).compute(real, synthetic),
            predictive_score=PredictiveScore(settings.predictor_training).compute(real, synthetic),
            marginal_distance=MarginalDistance(settings.num_quantiles).compute(real, synthetic),
            autocorrelation_distance=AutocorrelationDistance(settings.max_lag).compute(real, synthetic),
        )
